"""
core/generator.py
=================
Генерирует ответ через vLLM-сервер (OpenAI-совместимый API).
Гиперпараметры и системный промпт читаются из config.yaml.
Пути и адреса сервисов читаются из .env.
"""

import logging
import os
from pathlib import Path

import requests
import yaml
from dataclasses import dataclass
from typing import List

from core.retriever import RetrievedChunk

log = logging.getLogger(__name__)

# ── Загрузка config.yaml ──────────────────────────────────────────────────────
def _load_config() -> dict:
    config_path = Path(os.getenv("CONFIG_PATH", "/app/config.yaml"))
    if not config_path.exists():
        # fallback: ищем рядом с этим файлом (для локального запуска)
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

_config = _load_config()

# ── Настройки vLLM-сервера (из .env) ─────────────────────────────────────────
VLLM_URL     = os.getenv("VLLM_URL",     "http://vllm:8000/v1/chat/completions")
VLLM_MODEL   = os.getenv("VLLM_MODEL",   "/model")
VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "120"))

# ── Параметры генерации (из config.yaml) ──────────────────────────────────────
_gen = _config.get("generation", {})
MAX_NEW_TOKENS     = _gen.get("max_new_tokens",     300)
TEMPERATURE        = _gen.get("temperature",        0.01)
REPETITION_PENALTY = _gen.get("repetition_penalty", 1.1)

# ── Системный промпт (из config.yaml) ─────────────────────────────────────────
SYSTEM_PROMPT = _config.get("system_prompt", "Ты — корпоративный ассистент. Отвечай строго на основе источников.").strip()


@dataclass
class GenerationResult:
    answer:   str
    sources:  List[dict]
    n_chunks: int


class Generator:
    """
    Генератор ответов через vLLM HTTP API.
    Не загружает модель локально — только отправляет запросы к контейнеру.
    """

    def __init__(self):
        self._check_server()

    def _check_server(self):
        health_url = VLLM_URL.replace("/v1/chat/completions", "/health")
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                log.info(f"[Generator] vLLM сервер доступен: {VLLM_URL}")
            else:
                log.warning(f"[Generator] vLLM вернул статус {r.status_code}")
        except requests.exceptions.ConnectionError:
            log.error(
                f"[Generator] vLLM недоступен по адресу {VLLM_URL}. "
                "Убедитесь что контейнер запущен."
            )

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            header = f"[Источник {i}] {c.title}"
            if c.heading:
                header += f" → {c.heading}"
            if c.metadata.get("last_modified"):
                header += f" (изм. {c.metadata['last_modified']})"
            if c.metadata.get("author"):
                header += f" | автор: {c.metadata['author']}"
            breadcrumbs = c.metadata.get("breadcrumbs", [])
            if len(breadcrumbs) > 2:
                header += f"\nРаздел: {' > '.join(breadcrumbs[-3:])}"
            parts.append(f"{header}\n{c.text}")
        return "\n\n---\n\n".join(parts)

    def _call_vllm(self, system: str, user: str) -> str:
        payload = {
            "model":              VLLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens":         MAX_NEW_TOKENS,
            "temperature":        TEMPERATURE,
            "repetition_penalty": REPETITION_PENALTY,
        }
        response = requests.post(VLLM_URL, json=payload, timeout=VLLM_TIMEOUT)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def generate(self, question: str, chunks: List[RetrievedChunk]) -> GenerationResult:
        if not chunks:
            return GenerationResult(
                answer   = "В базе знаний нет информации по этому вопросу.",
                sources  = [],
                n_chunks = 0,
            )

        context  = self._format_context(chunks)
        user_msg = f"ИСТОЧНИКИ:\n{context}\n\nВОПРОС: {question}"

        try:
            answer = self._call_vllm(SYSTEM_PROMPT, user_msg)
        except requests.exceptions.Timeout:
            log.error("[Generator] Таймаут запроса к vLLM")
            answer = "Ошибка: сервер генерации не ответил в срок. Попробуйте повторить запрос."
        except requests.exceptions.RequestException as e:
            log.error(f"[Generator] Ошибка запроса к vLLM: {e}")
            answer = "Ошибка: не удалось получить ответ от сервера генерации."

        sources = [
            {
                "title"        : c.title,
                "heading"      : c.heading,
                "url"          : c.metadata.get("url", ""),
                "source_folder": c.source_folder,
                "last_modified": c.metadata.get("last_modified", ""),
                "rerank_score" : round(c.rerank_score, 3),
            }
            for c in chunks
        ]

        log.info(f"[Generator] Ответ получен ({len(answer)} символов)")
        return GenerationResult(answer=answer, sources=sources, n_chunks=len(chunks))
