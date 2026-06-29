"""
scripts/download.py
===================
Скачивает все модели RAG-чата из Hugging Face в локальные папки.

Запуск:
    python scripts/download.py
    python scripts/download.py --force    # перекачать всё заново
    python scripts/download.py --dry-run  # показать что будет скачано

Пути берутся из .env (MODELS_HOST_PATH, QWEN_MODEL_PATH).
HF_TOKEN используется для авторизации если задан.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download

load_dotenv()

REQUIRED_MODELS = {
    "multilingual-e5-large": {
        "repo_id": "intfloat/multilingual-e5-large",
        "env_key": "MODELS_HOST_PATH",
        "subdir": "multilingual-e5-large",
        "description": "Dense-эмбеддинги (1024-d)",
    },
    "bge-reranker-v2-m3": {
        "repo_id": "BAAI/bge-reranker-v2-m3",
        "env_key": "MODELS_HOST_PATH",
        "subdir": "bge-reranker-v2-m3",
        "description": "Cross-encoder реранкер",
    },
    "qwen2.5-7b-instruct": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "env_key": "QWEN_MODEL_PATH",
        "subdir": None,
        "description": "LLM для генерации ответов (через vLLM)",
    },
}


def resolve_paths() -> dict[str, Path]:
    models_host = os.getenv("MODELS_HOST_PATH", "data/models")
    qwen_path = os.getenv("QWEN_MODEL_PATH", "data/models/qwen2.5-7b-instruct-awq")

    paths = {}
    for name, info in REQUIRED_MODELS.items():
        if info["subdir"]:
            paths[name] = Path(models_host) / info["subdir"]
        else:
            paths[name] = Path(qwen_path)
    return paths


def auth_huggingface() -> str | None:
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("HF_TOKEN найден, авторизация в Hugging Face...")
        login(token=hf_token)
    else:
        print("HF_TOKEN не задан — анонимная загрузка (ограничения по скорости).")
    return hf_token


def download_models(force: bool = False, dry_run: bool = False) -> None:
    paths = resolve_paths()
    auth_huggingface()

    project_root = Path.cwd()
    os.environ["HF_HOME"] = str(project_root / "data" / "models" / "hf_cache")

    for name, info in REQUIRED_MODELS.items():
        target = paths[name]
        exists = target.exists() and any(target.iterdir()) if target.exists() else False

        if exists and not force:
            print(f"  [пропуск] {name} — уже скачана ({target})")
            continue

        if dry_run:
            print(f"  [dry-run] {name} -> {target} ({info['description']})")
            continue

        target.mkdir(parents=True, exist_ok=True)
        print(f"\n  Загрузка {info['repo_id']} -> {target}")
        print(f"  ({info['description']})")

        snapshot_download(
            repo_id=info["repo_id"],
            local_dir=str(target),
            local_dir_use_symlinks=False,
            max_workers=8,
        )
        print(f"  Готово: {target}")

    if dry_run:
        print("\n[dry-run] Ничего не скачано.")
    else:
        print("\nВсе модели успешно загружены.")


def main():
    parser = argparse.ArgumentParser(description="Загрузка моделей RAG-чата из Hugging Face")
    parser.add_argument("--force", action="store_true", help="Перекачать модели даже если они уже есть")
    parser.add_argument("--dry-run", action="store_true", help="Показать что будет скачано без загрузки")
    args = parser.parse_args()

    download_models(force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
