"""
scripts/sync_confluence.py
=========================
Загрузка страниц из Confluence и индексация в Qdrant.

Режимы работы:
  1. Автоматический (по расписанию): запускается раз в неделю в 2 ночи по МСК (воскресенье)
  2. Ручной: python scripts/sync_confluence.py
  3. Только загрузка JSON: python scripts/sync_confluence.py --download-only
  4. Только индексация (из JSON): python scripts/sync_confluence.py --index-only

Формат JSON аналогичен knowledge_base_cleaned.json:
  {
    "meta": { "exported_at": "...", "sources": ["confluence"], "total_docs": N },
    "documents": [ { "id": "...", "title": "...", "sections": [...], ... } ]
  }

Аутентификация (OAuth 2.0):
  1. Запустите scripts/auth_confluence.py для первичной авторизации
  2. Токены сохраняются в .confluence_tokens.json
  3. Access token обновляется автоматически

Переменные окружения (из .env):
  CONFLUENCE_URL           — базовый URL Confluence (например, https://confluence.example.com)
  CONFLUENCE_CLIENT_ID     — ID приложения (OAuth 2.0)
  CONFLUENCE_CLIENT_SECRET — Секрет приложения (OAuth 2.0)
  CONFLUENCE_SPACES        — список ключей пространств через запятую (опционально, по умолчанию все)
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
import yaml
from dotenv import load_dotenv
from html.parser import HTMLParser

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Настройки Confluence (из .env) ────────────────────────────────────────────
CONFLUENCE_URL           = os.getenv("CONFLUENCE_URL", "")
CONFLUENCE_CLIENT_ID     = os.getenv("CONFLUENCE_CLIENT_ID", "")
CONFLUENCE_CLIENT_SECRET = os.getenv("CONFLUENCE_CLIENT_SECRET", "")
CONFLUENCE_SPACES        = os.getenv("CONFLUENCE_SPACES", "")

# ── Пути ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
KB_FILE        = DATA_DIR / "knowledge_base.json"
SNAPSHOT_DIR   = DATA_DIR / "qdrant_snapshots"

# ── Загрузка config.yaml ──────────────────────────────────────────────────────
def _load_config() -> dict:
    config_path = Path(os.getenv("CONFIG_PATH", str(PROJECT_ROOT / "config.yaml")))
    if not config_path.exists():
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── HTML to plain text converter ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Очищает текст от артефактов и спецсимволов."""
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r'nbsp[b]?', ' ', text)
    text = re.sub(r'_{3,}', ' ', text)
    text = re.sub(r'«\s*»', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[A-Z]:\\(?:[\w\d\s]+\\)*[\w\d\s\.]+', '', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


class HTMLTextExtractor(HTMLParser):
    """Простой экстрактор текста из HTML."""

    SKIP_TAGS = {"script", "style", "code", "pre"}

    def __init__(self):
        super().__init__()
        self._text_parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._text_parts.append("\n")
        if tag == "td":
            self._text_parts.append(" | ")

    def handle_endtag(self, tag: str):
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "table"):
            self._text_parts.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            self._text_parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._text_parts)
        raw = re.sub(r'\n{3,}', '\n\n', raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    """Конвертирует HTML в plain text."""
    if not html:
        return ""
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        return html
    return clean_text(extractor.get_text())


# ── Confluence API Client ─────────────────────────────────────────────────────

class ConfluenceClient:
    """Клиент для Confluence REST API v1."""

    def __init__(self, base_url: str, oauth_client=None):
        self.base_url = base_url.rstrip("/")
        self.oauth = oauth_client
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    def _get(self, path: str, params: dict | None = None) -> dict:
        if self.oauth:
            token = self.oauth.get_valid_token()
            self.session.headers["Authorization"] = f"Bearer {token}"
        url = f"{self.base_url}/rest/api{path}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_spaces(self, limit: int = 100) -> list[dict]:
        """Получает список пространств."""
        spaces = []
        start = 0
        while True:
            data = self._get("/space", {"start": start, "limit": limit})
            results = data.get("results", [])
            spaces.extend(results)
            if len(results) < limit:
                break
            start += limit
        return spaces

    def get_content(
        self,
        space_key: str | None = None,
        content_type: str = "page",
        limit: int = 25,
        expand: str = "body.storage,version,space,ancestors",
    ) -> list[dict]:
        """Получает список контента (страниц) с пагинацией."""
        items = []
        start = 0
        while True:
            params: dict = {
                "type": content_type,
                "status": "current",
                "expand": expand,
                "start": start,
                "limit": limit,
            }
            if space_key:
                params["spaceKey"] = space_key

            data = self._get("/content", params)
            results = data.get("results", [])
            items.extend(results)

            log.info(
                f"[Confluence] Загружено {len(items)} "
                f"(space={space_key or 'all'}, limit={limit})"
            )

            if len(results) < limit:
                break
            start += limit
            time.sleep(0.2)

        return items

    def get_content_by_id(self, content_id: str, expand: str = "body.storage,version,space,ancestors") -> dict:
        """Получает конкретную страницу по ID."""
        return self._get(f"/content/{content_id}", {"expand": expand})

    def test_connection(self) -> bool:
        """Проверяет подключение к Confluence."""
        try:
            data = self._get("/content", {"limit": 1})
            log.info(f"[Confluence] Подключение успешно. Контент найден: {data.get('size', 0)}")
            return True
        except Exception as e:
            log.error(f"[Confluence] Ошибка подключения: {e}")
            return False


# ── Конвертация Confluence → формат knowledge_base ───────────────────────────

def _build_breadcrumbs(ancestors: list[dict]) -> list[str]:
    """Строит цепочку breadcrumbs из списка предков."""
    breadcrumbs = []
    for ancestor in ancestors:
        title = ancestor.get("title", "")
        if title:
            breadcrumbs.append(title)
    return breadcrumbs


def _extract_sections_from_storage(storage_body: str) -> list[dict]:
    """
    Извлекает секции из body.storage (HTML).
    Разбивает по заголовкам h1-h6.
    """
    text = html_to_text(storage_body)
    if not text.strip():
        return []

    lines = text.split("\n")
    sections: list[dict] = []
    current_heading: str | None = None
    current_text_parts: list[str] = []

    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)', re.MULTILINE)

    parts = heading_pattern.split(text)

    if len(parts) <= 1:
        return [{"heading": None, "text": text.strip()}]

    i = 0
    while i < len(parts):
        part = parts[i]
        if re.match(r'^#{1,6}$', part):
            if current_text_parts:
                accumulated = "\n".join(current_text_parts).strip()
                if accumulated:
                    sections.append({"heading": current_heading, "text": accumulated})
            current_heading = parts[i + 1].strip() if i + 1 < len(parts) else None
            current_text_parts = []
            i += 2
        else:
            current_text_parts.append(part)
            i += 1

    if current_text_parts:
        accumulated = "\n".join(current_text_parts).strip()
        if accumulated:
            sections.append({"heading": current_heading, "text": accumulated})

    return sections if sections else [{"heading": None, "text": text.strip()}]


def _extract_links_from_storage(storage_body: str) -> dict:
    """Извлекает ссылки из body.storage."""
    links: dict = {
        "external": [],
        "internal_page": [],
        "confluence_internal": [],
        "attachment_doc": [],
    }

    link_pattern = re.compile(
        r'<ac:link>.*?<ri:page ri:content-id="(\d+)".*?</ac:link>',
        re.DOTALL,
    )
    for match in link_pattern.finditer(storage_body):
        page_id = match.group(1)
        links["internal_page"].append({"href": f"{page_id}.html", "title": f"Page {page_id}"})

    href_pattern = re.compile(r'<a\s+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
    for match in href_pattern.finditer(storage_body):
        href, title = match.group(1), html_to_text(match.group(2))
        if href.startswith("http"):
            if "confluence.example.com" in href:
                links["confluence_internal"].append({"href": href, "title": title})
            else:
                links["external"].append({"href": href, "title": title})

    attachment_pattern = re.compile(
        r'<ac:link>.*?<ri:attachment ri:filename="([^"]+)".*?</ac:link>',
        re.DOTALL,
    )
    for match in attachment_pattern.finditer(storage_body):
        filename = match.group(1)
        links["attachment_doc"].append({"href": f"attachments/{filename}", "title": filename})

    attachment_link_pattern = re.compile(
        r'<a\s+href="([^"]*attachments/[^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    for match in attachment_link_pattern.finditer(storage_body):
        href, title = match.group(1), html_to_text(match.group(2))
        links["attachment_doc"].append({"href": href, "title": title})

    return links


def _extract_images_from_storage(storage_body: str) -> list[dict]:
    """Извлекает изображения из body.storage."""
    images = []
    pattern = re.compile(r'<ac:image[^>]*>.*?<ri:attachment ri:filename="([^"]+)".*?</ac:image>', re.DOTALL)
    for match in pattern.finditer(storage_body):
        filename = match.group(1)
        images.append({"src": f"attachments/{filename}", "alt": ""})
    return images


def convert_page_to_document(page: dict, source_folder: str = "confluence") -> dict:
    """
    Конвертирует страницу Confluence в формат документа knowledge_base.
    """
    page_id = page.get("id", "")
    title = page.get("title", "Untitled")
    space_key = page.get("space", {}).get("key", "")

    version_info = page.get("version", {})
    author_info = version_info.get("by", {})
    author = author_info.get("displayName", "")
    last_modified = version_info.get("when", "")[:10]

    ancestors = page.get("ancestors", [])
    breadcrumbs = _build_breadcrumbs(ancestors)
    breadcrumbs.insert(0, space_key)

    storage_body = ""
    body_data = page.get("body", {})
    if "storage" in body_data:
        storage_body = body_data["storage"].get("value", "")

    full_text = html_to_text(storage_body)
    sections = _extract_sections_from_storage(storage_body)
    links = _extract_links_from_storage(storage_body)
    images = _extract_images_from_storage(storage_body)

    doc_id = f"{source_folder}_{page_id}"
    page_url = f"{CONFLUENCE_URL}/pages/viewpage.action?pageId={page_id}"

    return {
        "id": doc_id,
        "url": page_url,
        "source_folder": source_folder,
        "filename": f"{page_id}.html",
        "title": title,
        "breadcrumbs": breadcrumbs,
        "author": author,
        "last_modified": last_modified,
        "full_text": full_text,
        "sections": sections,
        "links": links,
        "images": images,
    }


# ── Загрузка из Confluence ───────────────────────────────────────────────────

def download_from_confluence(spaces: list[str] | None = None) -> Path:
    """
    Загружает все страницы из Confluence и сохраняет в JSON.
    Возвращает путь к JSON-файлу.
    """
    if not CONFLUENCE_URL:
        log.error("[Confluence] CONFLUENCE_URL не задан в .env")
        sys.exit(1)

    oauth = None
    if CONFLUENCE_CLIENT_ID:
        from confluence_auth import ConfluenceOAuth2
        oauth = ConfluenceOAuth2(
            client_id=CONFLUENCE_CLIENT_ID,
            client_secret=CONFLUENCE_CLIENT_SECRET,
            base_url=CONFLUENCE_URL,
        )
        log.info("[Confluence] Используется OAuth 2.0 аутентификация")
    else:
        log.info("[Confluence] Используется анонимный доступ (client_id не задан)")

    client = ConfluenceClient(CONFLUENCE_URL, oauth_client=oauth)

    if not client.test_connection():
        log.error("[Confluence] Не удалось подключиться к Confluence")
        sys.exit(1)

    target_spaces = spaces or [s.strip() for s in CONFLUENCE_SPACES.split(",") if s.strip()]

    if not target_spaces:
        log.info("[Confluence] Пространства не указаны, загружаем все")
        all_spaces = client.get_spaces()
        target_spaces = [s["key"] for s in all_spaces]
        log.info(f"[Confluence] Найдено {len(target_spaces)} пространств: {target_spaces}")

    all_documents = []

    for space_key in target_spaces:
        log.info(f"[Confluence] Загрузка пространства: {space_key}")
        try:
            pages = client.get_content(space_key=space_key, limit=100)
            for page in pages:
                doc = convert_page_to_document(page, source_folder="confluence")
                all_documents.append(doc)
            log.info(f"[Confluence] Пространство {space_key}: {len(pages)} страниц")
        except Exception as e:
            log.error(f"[Confluence] Ошибка загрузки пространства {space_key}: {e}")
            continue

    if not all_documents:
        log.warning("[Confluence] Не загружено ни одного документа")
        sys.exit(1)

    kb_data = {
        "meta": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "sources": list(set(doc["source_folder"] for doc in all_documents)),
            "total_docs": len(all_documents),
        },
        "documents": all_documents,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)

    log.info(f"[Confluence] Сохранено {len(all_documents)} документов в {KB_FILE}")
    return KB_FILE


# ── Снапшот Qdrant ───────────────────────────────────────────────────────────

def create_snapshot(client) -> str | None:
    """Создаёт снапшот текущей коллекции Qdrant."""
    from qdrant_client.models import SnapshotDescription

    collection_name = os.getenv("COLLECTION_NAME", "acme_knowledge")

    try:
        if not client.collection_exists(collection_name):
            log.info("[Snapshot] Коллекция не существует, снапшот не нужен")
            return None

        info = client.get_collection(collection_name)
        if info.points_count == 0:
            log.info("[Snapshot] Коллекция пуста, снапшот не нужен")
            return None

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_name = f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        log.info(f"[Snapshot] Создание снапшота: {snapshot_name} ({info.points_count} точек)")
        client.create_snapshot(collection_name=collection_name)
        snapshots = client.list_snapshots(collection_name=collection_name)

        if snapshots:
            latest = snapshots[-1]
            log.info(f"[Snapshot] Снапшот создан: {latest.name}")
            return latest.name
        else:
            log.warning("[Snapshot] Снапшот создан, но не найден в списке")
            return None
    except Exception as e:
        log.warning(f"[Snapshot] Не удалось создать снапшот (пропускаем): {e}")
        return None


# ── Индексация в Qdrant ─────────────────────────────────────────────────────

def index_to_qdrant(kb_file: Path, recreate: bool = True):
    """Индексирует JSON-файл базы знаний в Qdrant."""
    import torch
    from tqdm import tqdm
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, SparseVectorParams,
        SparseIndexParams, PointStruct, SparseVector,
    )
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import uuid
    import re

    _config = _load_config()

    QDRANT_URL         = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME    = os.getenv("COLLECTION_NAME", "acme_knowledge")
    DENSE_VECTOR_NAME  = os.getenv("DENSE_VECTOR_NAME", "dense")
    SPARSE_VECTOR_NAME = os.getenv("SPARSE_VECTOR_NAME", "sparse")
    DENSE_MODEL_PATH   = os.getenv("DENSE_MODEL_PATH", "/app/models/multilingual-e5-large")
    SPARSE_MODEL_NAME  = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")

    _idx = _config.get("indexing", {})
    CHUNK_SIZE        = _idx.get("chunk_size", 1000)
    CHUNK_OVERLAP     = _idx.get("chunk_overlap", 300)
    BATCH_SIZE        = _idx.get("batch_size", 64)
    DENSE_VECTOR_SIZE = _idx.get("dense_vector_size", 1024)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    def make_chunks(doc: dict) -> list[dict]:
        chunks = []
        sections = doc.get("sections") or [{"heading": None, "text": doc.get("full_text", "")}]
        bc_context = " > ".join(doc.get("breadcrumbs", [])[-3:])

        for sec in sections:
            raw_text = sec.get("text", "").strip()
            heading  = sec.get("heading") or ""
            if not raw_text:
                continue

            if any(word in heading.lower() for word in ["вопрос", "faq", "ответ"]):
                sub_sections = re.split(r'(?m)^В:', raw_text)
                sub_sections = ["В:" + s for s in sub_sections if s.strip()]
            else:
                sub_sections = [raw_text]

            for sub_sec_text in sub_sections:
                for i, sub in enumerate(splitter.split_text(sub_sec_text)):
                    embed_text = (
                        f"passage: {bc_context} | {doc['title']} | {heading} | {sub}"
                        .replace("  ", " ").strip()
                    )
                    rag_text = f"Источник: {doc['title']}\nРаздел: {bc_context}\n"
                    if heading:
                        rag_text += f"Тема: {heading}\n"
                    rag_text += f"---\n{sub}"

                    chunks.append({
                        "embed_text": embed_text,
                        "rag_text":   rag_text,
                        "payload":    {
                            "doc_id":        doc["id"],
                            "url":           doc.get("url", ""),
                            "title":         doc["title"],
                            "heading":       heading,
                            "breadcrumbs":   doc.get("breadcrumbs", []),
                            "source_folder": doc.get("source_folder", ""),
                            "text":          rag_text,
                            "chunk_index":   i,
                        },
                    })
        return chunks

    def encode_sparse_batch(sparse_model, texts: list) -> list:
        return [
            SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
            for emb in sparse_model.embed(texts)
        ]

    def chunks_batched(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    log.info(f"[Index] Загружаем базу знаний: {kb_file}")
    with open(kb_file, encoding="utf-8") as f:
        kb = json.load(f)
    documents = kb["documents"]
    log.info(f"[Index] Документов: {len(documents)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[Index] Устройство: {device}")

    log.info(f"[Index] Загружаем dense модель: {DENSE_MODEL_PATH}")
    dense_model = SentenceTransformer(DENSE_MODEL_PATH, device=device)

    log.info(f"[Index] Загружаем sparse модель: {SPARSE_MODEL_NAME}")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    log.info(f"[Index] Подключаемся к Qdrant: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, timeout=120)

    if client.collection_exists(COLLECTION_NAME):
        if recreate:
            log.info(f"[Index] Удаляем существующую коллекцию '{COLLECTION_NAME}'...")
            try:
                client.delete_collection(COLLECTION_NAME)
                log.info(f"[Index] Коллекция удалена")
            except Exception as e:
                log.warning(f"[Index] Не удалось удалить коллекцию (продолжаем): {e}")
        else:
            log.info(
                f"[Index] Коллекция '{COLLECTION_NAME}' существует. "
                "Режим upsert."
            )

    if not client.collection_exists(COLLECTION_NAME):
        log.info(f"[Index] Создаём коллекцию '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )

    log.info("[Index] Создаём чанки...")
    all_chunks = []
    for doc in tqdm(documents, desc="Preprocessing"):
        all_chunks.extend(make_chunks(doc))
    log.info(f"[Index] Всего чанков: {len(all_chunks)}")

    batches = list(chunks_batched(all_chunks, BATCH_SIZE))
    for batch in tqdm(batches, desc="Uploading"):
        dense_vecs = dense_model.encode(
            [c["embed_text"] for c in batch],
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
        )
        sparse_vecs = encode_sparse_batch(sparse_model, [c["rag_text"] for c in batch])

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR_NAME:  dv.tolist(),
                    SPARSE_VECTOR_NAME: sv,
                },
                payload=chunk["payload"],
            )
            for dv, sv, chunk in zip(dense_vecs, sparse_vecs, batch)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    info = client.get_collection(COLLECTION_NAME)
    log.info(f"[Index] Готово! Поинтов в коллекции: {info.points_count}")


# ── Основной пайплайн ────────────────────────────────────────────────────────

def run_full_pipeline(spaces: list[str] | None = None, recreate: bool = True):
    """
    Полный пайплайн: загрузка → снапшот → индексация → удаление JSON.
    """
    log.info("=" * 60)
    log.info("[Pipeline] Начало синхронизации с Confluence")
    log.info("=" * 60)

    kb_file = download_from_confluence(spaces)

    from qdrant_client import QdrantClient
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=QDRANT_URL)
    snapshot_name = create_snapshot(client)

    if snapshot_name:
        log.info(f"[Pipeline] Снапшот создан: {snapshot_name}")

    index_to_qdrant(kb_file, recreate=recreate)

    try:
        kb_file.unlink()
        log.info(f"[Pipeline] JSON-файл удалён: {kb_file}")
    except Exception as e:
        log.warning(f"[Pipeline] Не удалось удалить JSON: {e}")

    log.info("[Pipeline] Синхронизация завершена")


def run_index_only(kb_file: Path, recreate: bool = True):
    """Только индексация из JSON (без загрузки из Confluence)."""
    if not kb_file.exists():
        log.error(f"[Index] Файл не найден: {kb_file}")
        sys.exit(1)

    from qdrant_client import QdrantClient
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=QDRANT_URL, timeout=120)
    snapshot_name = create_snapshot(client)

    if snapshot_name:
        log.info(f"[Pipeline] Снапшот создан: {snapshot_name}")

    index_to_qdrant(kb_file, recreate=recreate)

    try:
        kb_file.unlink()
        log.info(f"[Pipeline] JSON-файл удалён: {kb_file}")
    except Exception as e:
        log.warning(f"[Pipeline] Не удалось удалить JSON: {e}")

    log.info("[Pipeline] Индексация завершена")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Синхронизация Confluence → Qdrant"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Только загрузить из Confluence в JSON (без индексации)",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Только проиндексировать существующий JSON в Qdrant",
    )
    parser.add_argument(
        "--kb",
        type=Path,
        default=KB_FILE,
        help=f"Путь к JSON-файлу (по умолчанию: {KB_FILE})",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Не пересоздавать коллекцию (режим upsert)",
    )
    parser.add_argument(
        "--spaces",
        type=str,
        default="",
        help="Ключи пространств через запятую (перекрывает CONFLUENCE_SPACES)",
    )
    args = parser.parse_args()

    recreate = not args.no_recreate
    spaces = [s.strip() for s in args.spaces.split(",") if s.strip()] or None

    if args.download_only:
        download_from_confluence(spaces)
    elif args.index_only:
        run_index_only(args.kb, recreate=recreate)
    else:
        run_full_pipeline(spaces, recreate=recreate)


if __name__ == "__main__":
    main()
