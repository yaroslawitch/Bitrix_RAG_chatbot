"""
core/retriever.py — Гибридный поиск + Reranking
=================================================
Шаг 1: Dense (multilingual-e5-large) + Sparse (Qdrant/bm25) → RRF fusion → кандидаты
Шаг 2: Cross-Encoder (bge-reranker-v2-m3) → переранжирование → топ-N

Гиперпараметры читаются из config.yaml.
Пути к моделям и адреса сервисов читаются из .env.
"""

import logging
import os
import time
from pathlib import Path

import torch
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector,
    Prefetch, FusionQuery, Fusion,
)

log = logging.getLogger(__name__)

def _sanitize_text(text: str) -> str:
    """Убирает битые surrogate-символы и нормализует пробелы."""
    if not isinstance(text, str):
        text = str(text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return " ".join(text.split())

# ── Загрузка config.yaml ──────────────────────────────────────────────────────
def _load_config() -> dict:
    config_path = Path(os.getenv("CONFIG_PATH", "/app/config.yaml"))
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

_config = _load_config()

# ── Пути к моделям и сервисам (из .env) ──────────────────────────────────────
DENSE_MODEL_PATH   = os.getenv("DENSE_MODEL_PATH",   "/app/models/multilingual-e5-large")
RERANK_MODEL_PATH  = os.getenv("RERANK_MODEL_PATH",  "/app/models/bge-reranker-v2-m3")
SPARSE_MODEL_NAME  = os.getenv("SPARSE_MODEL_NAME",  "Qdrant/bm25")
QDRANT_URL         = os.getenv("QDRANT_URL",         "http://qdrant:6333")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME",    "acme_knowledge")
DENSE_VECTOR_NAME  = os.getenv("DENSE_VECTOR_NAME",  "dense")
SPARSE_VECTOR_NAME = os.getenv("SPARSE_VECTOR_NAME", "sparse")

# ── Гиперпараметры поиска (из config.yaml) ────────────────────────────────────
_ret = _config.get("retrieval", {})
PREFETCH_LIMIT    = _ret.get("prefetch_limit",    20)
RERANK_CANDIDATES = _ret.get("rerank_candidates", 20)
FINAL_TOP_K       = _ret.get("final_top_k",        3)
RERANK_THRESHOLD  = _ret.get("rerank_threshold",  -2.0)


# ── Датакласс результата ───────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text:          str
    title:         str
    heading:       str
    source_folder: str
    score:         float
    rerank_score:  float = 0.0
    sub_question:  Optional[str] = None
    metadata:      Dict[str, Any] = field(default_factory=dict)


# ── Retriever ─────────────────────────────────────────────────────────────────

class HybridRetriever:
    """Гибридный ретривер: dense + sparse → RRF → reranking."""

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"[Retriever] Устройство: {device}")

        log.info(f"[Retriever] (1/3) Загрузка dense модели: {DENSE_MODEL_PATH}")
        t = time.time()
        self.dense_model = SentenceTransformer(DENSE_MODEL_PATH, device=device)
        log.info(f"[Retriever] Dense готова за {time.time()-t:.1f}с")

        log.info(f"[Retriever] (2/3) Загрузка sparse: {SPARSE_MODEL_NAME}")
        t = time.time()
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        log.info(f"[Retriever] Sparse готова за {time.time()-t:.1f}с")

        log.info(f"[Retriever] (3/3) Загрузка reranker: {RERANK_MODEL_PATH}")
        t = time.time()
        self.reranker = CrossEncoder(RERANK_MODEL_PATH, device=device, max_length=512)
        log.info(f"[Retriever] Reranker готов за {time.time()-t:.1f}с")

        self.client = QdrantClient(url=QDRANT_URL)
        log.info(f"[Retriever] Готов. Коллекция: {COLLECTION_NAME}")

    # ── Внутренние методы ─────────────────────────────────────────────────────

    def _encode_sparse(self, text: str) -> SparseVector:
        """Кодирует текст в sparse вектор через BM25."""
        emb = list(self.sparse_model.embed([text]))[0]
        return SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )

    def _hybrid_search(self, query: str, limit: int) -> List[RetrievedChunk]:
        """Dense + sparse → RRF. Возвращает limit кандидатов."""

        query = _sanitize_text(query)
        if not query:
            return []

        # Dense эмбеддинг — prefix конкатенируется вручную (multilingual-e5-large)
        dense_vec = self.dense_model.encode(
            f"query: {query}",
            normalize_embeddings=True,
            convert_to_tensor=False,
        ).tolist()

        # Sparse эмбеддинг через BM25
        sparse_vec = self._encode_sparse(query)

        try:
            hits = self.client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(query=dense_vec,  using=DENSE_VECTOR_NAME,  limit=PREFETCH_LIMIT),
                    Prefetch(query=sparse_vec, using=SPARSE_VECTOR_NAME, limit=PREFETCH_LIMIT),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            ).points
        except Exception as e:
            log.error(f"[Retriever] Qdrant ошибка: {e}")
            return []

        results = []
        for h in hits:
            p = h.payload or {}
            results.append(RetrievedChunk(
                text          = p.get("text", ""),
                title         = p.get("title", ""),
                heading       = p.get("heading", ""),
                source_folder = p.get("source_folder", ""),
                score         = h.score,
                metadata      = p,
            ))
        return results

    def _rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Cross-Encoder переранжирует кандидатов и отсекает нерелевантные."""
        if not chunks:
            return []

        pairs  = [[query, c.text] for c in chunks]
        scores = self.reranker.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
        log.info(f"[Reranker] Scores: {[round(c.rerank_score, 3) for c in ranked]}")

        filtered = [c for c in ranked if c.rerank_score >= RERANK_THRESHOLD]
        result   = filtered[:top_k]

        dropped = len(ranked) - len(result)
        if dropped:
            log.info(f"[Reranker] Отсечено {dropped} чанков (threshold={RERANK_THRESHOLD})")

        return result

    # ── Публичный метод ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = FINAL_TOP_K) -> List[RetrievedChunk]:
        """
        Полный пайплайн для одного запроса: hybrid search → reranking.

        Args:
            query:  поисковый запрос (один подвопрос)
            top_k:  максимум чанков в результате

        Returns:
            List[RetrievedChunk], отсортированный по rerank_score убыванию
        """
        candidates = self._hybrid_search(query, limit=RERANK_CANDIDATES)
        log.info(f"[Retriever] '{query[:60]}' → {len(candidates)} кандидатов")

        results = self._rerank(query, candidates, top_k=top_k)
        log.info(
            f"[Retriever] Итог: {len(results)} чанков | "
            f"scores: {[round(r.rerank_score, 3) for r in results]}"
        )
        return results
