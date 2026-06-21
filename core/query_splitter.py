"""
core/query_splitter.py
======================
Детерминированный сплиттер запросов на основе правил — без LLM.

Почему не LLM:
  Qwen-7b нестабилен для этой задачи — периодически переключается на
  китайский язык, возвращает невалидный JSON, добавляет лишний текст.
  Детерминированный подход: мгновенно, предсказуемо, без GPU.

Алгоритм:
  1. Явное разделение по "?" — "Что такое LQA? Кто его проводит?"
  2. Разделение по союзам перед вопросительными словами/глаголами —
     "Что такое LQA и кто его проводит?"
  3. Раскрытие местоимений через привязку к последнему найденному существительному
  4. Если разбить не удалось — возвращает исходный запрос как есть

Использование из orchestrator.py:
    from core.query_splitter import QuerySplitter
    splitter = QuerySplitter()
    splitter.split("Что такое LQA и кто его проводит?")
    # → ("Что такое LQA?", "Кто проводит LQA?")
"""

import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# ── Вопросительные слова и глаголы — маркеры начала нового вопроса ────────────
QUESTION_WORDS = (
    "что", "кто", "где", "когда", "зачем", "почему", "как", "сколько",
    "какой", "какая", "какое", "какие", "чей", "чья", "чьё", "чьи",
    "можно", "нужно", "стоит", "является", "делать", "работает",
    "используется", "происходит", "проводит", "хранится", "находится",
    "отличается", "входит", "влияет", "означает",
)

# ── Местоимения которые нужно раскрывать ─────────────────────────────────────
PRONOUNS = {
    # мужской род
    "его", "ему", "им", "него", "нему", "ним", "них",
    # женский род
    "её", "ее", "ей", "ней",
    # средний/множественный
    "их", "ими", "ними",
    # указательные
    "это", "этого", "этому", "этим", "этой", "этих",
    "там", "туда", "оттуда", "тут", "здесь",
    # возвратные
    "им", "ими",
}

# ── Союзы-разделители составных запросов ─────────────────────────────────────
SPLIT_CONJUNCTIONS = (
    r"\bи\b",
    r"\bа также\b",
    r"\bплюс\b",
    r"\bещё\b",
    r"\bеще\b",
    r"\bа\b",
)


class QuerySplitter:
    """
    Детерминированный сплиттер запросов.
    Не использует LLM — только регулярные выражения и эвристики.
    """

    def __init__(self):
        # Паттерн для союзов-разделителей
        self._conjunction_pattern = re.compile(
            "(" + "|".join(SPLIT_CONJUNCTIONS) + r")\s+(" +
            "|".join(QUESTION_WORDS) + r")\b",
            re.IGNORECASE,
        )
        log.info("[QuerySplitter] Инициализирован (детерминированный режим)")

    # ── Раскрытие местоимений ─────────────────────────────────────────────────

    def _extract_subject(self, text: str) -> Optional[str]:
        """
        Извлекает главный субъект запроса — существительное или аббревиатуру.
        Используется для раскрытия местоимений во втором подвопросе.

        Стратегия: берём первое существительное/аббревиатуру после
        вопросительного слова в начале запроса.

        Примеры:
          "Что такое LQA и кто его проводит?" → "LQA"
          "Как работает SVN и как им пользоваться?" → "SVN"
          "Где хранятся документы и как их найти?" → "документы"
        """
        # Сначала ищем аббревиатуру (2-6 заглавных букв) — они почти всегда субъект
        abbr = re.search(r'\b[A-ZА-ЯЁ]{2,6}\b', text)
        if abbr:
            return abbr.group()

        # Ищем первое существительное после вопросительного слова.
        # Эвристика: слово длиннее 4 букв, не является вопросительным словом,
        # союзом или предлогом — скорее всего существительное.
        stopwords = set(QUESTION_WORDS) | {
            "такое", "это", "из", "в", "на", "по", "с", "к", "у",
            "за", "от", "до", "при", "об", "для", "без",
        }
        words = re.findall(r'\b[а-яёА-ЯЁa-zA-Z]{4,}\b', text)
        for word in words:
            if word.lower() not in stopwords:
                return word

        return None

    def _resolve_pronouns(self, text: str, subject: str) -> str:
        """
        Заменяет местоимения в тексте на субъект.

        Примеры:
          "Кто его проводит?" + subject="LQA" → "Кто проводит LQA?"
          "Как им пользоваться?" + subject="SVN" → "Как пользоваться SVN?"
          "Как их найти?" + subject="документы" → "Как найти документы?"
        """
        result = text

        # Строим паттерн для всех местоимений
        pronoun_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(p) for p in PRONOUNS) + r')\b',
            re.IGNORECASE,
        )

        # Заменяем местоимение на субъект
        result = pronoun_pattern.sub(subject, result)

        # Убираем двойные пробелы после замены
        result = re.sub(r'\s{2,}', ' ', result).strip()

        return result

    # ── Стратегии разбивки ────────────────────────────────────────────────────

    def _split_by_question_marks(self, text: str) -> Optional[tuple[str, ...]]:
        """
        Стратегия 1: разбивка по явным вопросительным знакам.
        "Что такое LQA? Кто его проводит?" → ("Что такое LQA?", "Кто его проводит?")
        """
        # Разбиваем по "?" с последующим пробелом и заглавной буквой
        parts = re.split(r'\?\s+(?=[А-ЯЁA-Z])', text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) < 2:
            return None

        # Восстанавливаем знаки вопроса
        result = []
        for p in parts:
            if not p.endswith("?"):
                p += "?"
            result.append(p)

        return tuple(result)

    def _split_by_conjunction(self, text: str) -> Optional[tuple[str, ...]]:
        """
        Стратегия 2: разбивка по союзу перед вопросительным словом/глаголом.
        "Что такое LQA и кто его проводит?" → ("Что такое LQA?", "Кто проводит LQA?")
        """
        match = self._conjunction_pattern.search(text)
        if not match:
            return None

        split_pos = match.start()

        first  = text[:split_pos].strip()
        second = text[match.start(2):].strip()  # начинаем со второго вопроса (без союза)

        # Добавляем "?" к первой части если нет
        if first and not first.endswith("?"):
            first += "?"

        # Добавляем "?" ко второй части если нет
        if second and not second.endswith("?"):
            second += "?"

        if not first or not second:
            return None

        # Раскрываем местоимения во втором подвопросе
        subject = self._extract_subject(first)
        if subject:
            second_resolved = self._resolve_pronouns(second, subject)
            log.debug(f"[QuerySplitter] Раскрытие местоимений: '{second}' → '{second_resolved}' (субъект: '{subject}')")
            second = second_resolved

        return (first, second)

    def _split_by_enumeration(self, text: str) -> Optional[tuple[str, ...]]:
        """
        Стратегия 3: разбивка по маркерам перечисления.
        "1) Что такое LQA? 2) Кто его проводит?" → ("Что такое LQA?", "Кто проводит LQA?")
        """
        # Паттерны: "1)", "2)", "а)", "б)", "во-первых", "во-вторых"
        enum_pattern = re.compile(
            r'(?:^|\s)(?:\d+\)|[а-я]\)|во[-\s]?первых|во[-\s]?вторых|в[-\s]?третьих)\s*',
            re.IGNORECASE,
        )
        parts = enum_pattern.split(text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) < 2:
            return None

        result = []
        subject = None
        for i, p in enumerate(parts):
            if not p.endswith("?"):
                p += "?"
            if i == 0:
                subject = self._extract_subject(p)
            elif subject:
                p = self._resolve_pronouns(p, subject)
            result.append(p)

        return tuple(result)

    # ── Публичный метод ───────────────────────────────────────────────────────

    def split(self, user_text: str) -> tuple[str, ...]:
        """
        Разбивает запрос пользователя на подвопросы.

        Пробует стратегии по приоритету:
          1. Разбивка по явным "?" (самый надёжный признак)
          2. Разбивка по союзу перед вопросительным словом
          3. Разбивка по маркерам перечисления
          4. Fallback: возвращает исходный запрос как есть

        Args:
            user_text: исходный запрос пользователя

        Returns:
            Кортеж строк — каждая строка самодостаточный вопрос.
            Если вопрос один — кортеж из одного элемента.

        Examples:
            split("Что такое LQA? Кто его проводит?")
            → ("Что такое LQA?", "Кто проводит LQA?")

            split("Что такое SVN и как им пользоваться?")
            → ("Что такое SVN?", "Как пользоваться SVN?")

            split("Каков срок сдачи проекта?")
            → ("Каков срок сдачи проекта?",)
        """
        text = user_text.strip()

        # Стратегия 1: явные вопросительные знаки
        result = self._split_by_question_marks(text)
        if result:
            log.info(f"[QuerySplitter] Стратегия 1 (по '?'): {len(result)} подвопроса")
            log.info(f"[QuerySplitter] '{text[:60]}' → {result}")
            return result

        # Стратегия 2: союз перед вопросительным словом
        result = self._split_by_conjunction(text)
        if result:
            log.info(f"[QuerySplitter] Стратегия 2 (по союзу): {len(result)} подвопроса")
            log.info(f"[QuerySplitter] '{text[:60]}' → {result}")
            return result

        # Стратегия 3: маркеры перечисления
        result = self._split_by_enumeration(text)
        if result:
            log.info(f"[QuerySplitter] Стратегия 3 (по нумерации): {len(result)} подвопроса")
            log.info(f"[QuerySplitter] '{text[:60]}' → {result}")
            return result

        # Fallback: одиночный запрос
        log.info(f"[QuerySplitter] Одиночный запрос: '{text[:60]}'")
        return (text,)