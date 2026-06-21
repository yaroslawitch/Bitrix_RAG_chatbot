import os
from pathlib import Path

from huggingface_hub import login, snapshot_download
from dotenv import load_dotenv


REQUIRED_MODELS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}


def resolve_paths() -> tuple[Path, Path]:
    project_root = Path.cwd()
    data_dir = project_root / "data"
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, models_dir


def auth_huggingface() -> str | None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        print("Обнаружен HF_TOKEN, выполняю авторизацию в Hugging Face...")
        login(token=hf_token)
    else:
        print(
            "HF_TOKEN не найден. Продолжаю анонимную загрузку "
            "(возможны ограничения по скорости)."
        )

    return hf_token


def download_models() -> None:
    _, models_dir = resolve_paths()
    hf_token = auth_huggingface()

    os.environ["HF_HOME"] = str(models_dir / "hf_cache")

    for local_name, repo_id in REQUIRED_MODELS.items():
        target_dir = models_dir / local_name
        print(f"\nНачинаю загрузку {repo_id} -> {target_dir}")

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            token=hf_token,
            local_dir_use_symlinks=False,
            max_workers=8,
        )

        print(f"Модель успешно сохранена в: {target_dir}")

    print("\nВсе обязательные модели успешно загружены.")


if __name__ == "__main__":
    download_models()
