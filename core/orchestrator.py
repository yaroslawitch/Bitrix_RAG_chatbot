# core/orchestrator.py
"""
core/orchestrator.py — Центральный оркестратор RAG-пайплайна
=============================================================
Инициализирует все компоненты и связывает их в единый пайплайн.

Порядок инициализации:
  1. Generator       — отвечает за финальную генерацию ответа (через vLLM)
  2. QuerySplitter   — детерминированно разбивает запрос на подвопросы (без LLM)
  3. HybridRetriever — ищет релевантные чанки

Пайплайн:
  splitter.split(query) → (q1,) или (q1, q2, ...)
  для каждого qi: retriever.search(qi) → chunks → generator.generate(qi, chunks)
  → OrchestratorResult (объединённый ответ + детализация по подвопросам)

Использование:
    from core.orchestrator import Orchestrator
    orch = Orchestrator()
    result = orch.answer("Что такое LQA и кто его проводит?")
    print(result.answer)
    print(result.used_splitter)   # True — запрос был разбит
    print(result.n_sub_questions) # 2
"""

import logging
from dataclasses import dataclass
from typing import List
from pathlib import Path
import yaml, os

from core.generator import Generator, GenerationResult
from core.retriever import HybridRetriever, RetrievedChunk
from core.query_splitter import QuerySplitter

log = logging.getLogger(__name__)

def _sanitize_text(text: str) -> str:
    """Нормализует пользовательский текст и убирает битые surrogate-символы."""
    if not isinstance(text, str):
        text = str(text)
    # Удаляем одиночные surrogate-коды (часто появляются из проблем кодировки консоли)
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return " ".join(text.split())


@dataclass
class SubAnswer:
    question: str
    answer:   str
    sources:  List[dict]
    n_chunks: int


@dataclass
class OrchestratorResult:
    original_query:  str
    sub_answers:     List[SubAnswer]
    answer:          str
    n_sub_questions: int
    used_splitter:   bool  # для отладки: гонялся ли запрос через сплиттер


class Orchestrator:
    """Центральный класс RAG-пайплайна."""

    def __init__(self):
        log.info("[Orchestrator] Запуск инициализации компонентов...")

        log.info("[Orchestrator] (1/3) Generator...")
        self.generator = Generator()

        log.info("[Orchestrator] (2/3) QuerySplitter...")
        self.splitter = QuerySplitter()

        log.info("[Orchestrator] (3/3) HybridRetriever...")
        self.retriever = HybridRetriever()

        log.info("[Orchestrator] Все компоненты готовы.")

        self._config = yaml.safe_load(open(
            os.getenv("CONFIG_PATH", "/app/config.yaml"), encoding="utf-8"
        ))

        self.TOP_K = self._config.get("orchestrator", {}).get("top_k", 5)

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _deduplicate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Убирает дублирующиеся чанки по тексту."""
        seen:   set[str] = set()
        result: List[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.text not in seen:
                seen.add(chunk.text)
                result.append(chunk)
        return result

    def _format_answer(self, sub_answers: List[SubAnswer]) -> str:
        """Один вопрос → ответ без заголовка. Несколько → нумерованный список."""
        if len(sub_answers) == 1:
            return sub_answers[0].answer

        parts = []
        for i, sa in enumerate(sub_answers, 1):
            parts.append(f"**{i}. {sa.question}**\n{sa.answer}")
        return "\n\n".join(parts)

    # ── Основной метод ────────────────────────────────────────────────────────

    def answer(self, user_query: str) -> OrchestratorResult:
        """
        Полный RAG-пайплайн: разбивка → поиск → генерация → объединение.

        Сплиттер детерминированный и мгновенный — всегда вызывается.
        Если вопрос одиночный, возвращает кортеж из одного элемента
        и пайплайн проходит без накладных расходов.

        Args:
            user_query: исходный запрос пользователя
            top_k:      максимум чанков на каждый подвопрос

        Returns:
            OrchestratorResult с итоговым ответом и детализацией
        """
        top_k = self.TOP_K

        clean_query = _sanitize_text(user_query)
        log.info(f"[Orchestrator] Запрос: '{clean_query}'")

        # ── Шаг 1: разбивка запроса ───────────────────────────────────────────
        sub_questions = self.splitter.split(clean_query)
        used_splitter = len(sub_questions) > 1
        log.info(f"[Orchestrator] Подвопросов: {len(sub_questions)} | Сплиттер сработал: {used_splitter}")

        # ── Шаг 2: поиск + генерация для каждого подвопроса ──────────────────
        sub_answers: List[SubAnswer] = []

        for sub_q in sub_questions:
            log.info(f"[Orchestrator] Подвопрос: '{sub_q}'")

            chunks: List[RetrievedChunk] = self.retriever.search(sub_q, top_k=top_k)
            chunks = self._deduplicate(chunks)

            gen_result: GenerationResult = self.generator.generate(sub_q, chunks)

            sub_answers.append(SubAnswer(
                question=sub_q,
                answer=gen_result.answer,
                sources=gen_result.sources,
                n_chunks=gen_result.n_chunks,
            ))

        # ── Шаг 3: объединение ────────────────────────────────────────────────
        combined = self._format_answer(sub_answers)

        result = OrchestratorResult(
            original_query=clean_query,
            sub_answers=sub_answers,
            answer=combined,
            n_sub_questions=len(sub_questions),
            used_splitter=used_splitter,
        )

        log.info(
            f"[Orchestrator] Готово. "
            f"Подвопросов: {result.n_sub_questions} | "
            f"Сплиттер: {used_splitter} | "
            f"Длина ответа: {len(combined)} символов"
        )
        return result