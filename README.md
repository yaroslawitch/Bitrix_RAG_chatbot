# RAG-ассистент на базе знаний

Корпоративный ассистент на основе RAG-пайплайна (Retrieval-Augmented Generation).  
Отвечает на вопросы сотрудников строго на основе загруженной базы знаний.

---

## Стек

|Компонент|Технология|
|-|-|
|LLM (генерация)|Qwen 2.5-7B AWQ через vLLM|
|Эмбеддинги (dense)|multilingual-e5-large|
|Эмбеддинги (sparse)|BM25 (fastembed)|
|Реранкинг|bge-reranker-v2-m3|
|Векторная БД|Qdrant|
|Оркестрация|Docker Compose|
|Источник данных|Confluence REST API|

---

## Структура проекта

```
RAG-Chatbot/
├── core/
│   ├── generator.py          # Генерация ответа через vLLM API
│   ├── retriever.py          # Гибридный поиск + реранкинг
│   ├── orchestrator.py       # Оркестратор пайплайна
│   └── query_splitter.py     # Детерминированный сплиттер запросов
├── scripts/
│   ├── sync_confluence.py    # Загрузка из Confluence + индексация в Qdrant
│   ├── confluence_auth.py    # OAuth 2.0 клиент для Confluence
│   ├── auth_confluence.py    # Скрипт первичной авторизации
│   ├── scheduler.py          # Планировщик синхронизации (по расписанию)
│   └── download_all_models.py # Скрипт скачивания моделей
├── api/
│   └── bitrix_webhook.py     # FastAPI сервер для Bitrix24
├── data/
│   ├── models/               # Папка с моделями (не в Docker-образе)
│   │   ├── multilingual-e5-large/
│   │   ├── bge-reranker-v2-m3/
│   │   └── qwen2.5-7b-instruct-awq/
│   └── qdrant_storage/       # Хранилище Qdrant (не в Docker-образе)
├── config.yaml               # Гиперпараметры и системный промпт
├── .env                      # Пути и адреса сервисов
├── docker-compose.yml        # Описание сервисов
├── Dockerfile                # Сборка app-контейнера
├── requirements.txt          # Python-зависимости
└── main.py                   # Точка входа (консольный чат)
```

\---

## Требования

* Docker Desktop (Windows) или Docker Engine (Linux)
* NVIDIA GPU с поддержкой CUDA 12.1+ (рекомендуется RTX 3090 или аналог)
* NVIDIA Container Toolkit (`nvidia-docker2`)
* Драйвер NVIDIA R525 или новее
* \~30GB свободного места на диске (модели + образы)

Проверить драйвер:

```bash
nvidia-smi
```

\---

## Конфигурация

### `.env` — пути и адреса сервисов

Файл `.env` лежит в корне проекта. Содержит пути которые зависят от конкретной машины.

```env
# Пути к моделям внутри Docker-контейнера (не менять)
DENSE_MODEL_PATH=/app/models/multilingual-e5-large
RERANK_MODEL_PATH=/app/models/bge-reranker-v2-m3
SPARSE_MODEL_NAME=Qdrant/bm25

# vLLM — адрес внутри Docker-сети (не менять)
VLLM_URL=http://vllm:8000/v1/chat/completions
VLLM_MODEL=/model
VLLM_TIMEOUT=120

# Qdrant — адрес внутри Docker-сети (не менять)
QDRANT_URL=http://qdrant:6333
COLLECTION_NAME=acme_knowledge
DENSE_VECTOR_NAME=dense
SPARSE_VECTOR_NAME=sparse

# Пути на хосте для маунтов (МЕНЯТЬ ПОД СВОЮ МАШИНУ)
QDRANT_STORAGE_PATH=/path/to/data/qdrant_storage
MODELS_HOST_PATH=/path/to/data/models
QWEN_MODEL_PATH=/path/to/data/models/qwen2.5-7b-instruct-awq

# Confluence API (опционально)
CONFLUENCE_URL=https://confluence.example.com
CONFLUENCE_SPACES=TST
```

**Важно про пути на Windows:**  
В Git Bash и Docker Compose пути Windows пишутся в Unix-формате:  
`D:\Projects\data\models` → `/d/Projects/data/models`

**Три нижних переменные** (`*_HOST_PATH`) — это маунты с хоста в контейнер.  
Они должны указывать на реальные папки на твоей машине.  
Остальные переменные — внутренние адреса Docker-сети, менять не нужно.

\---

### `config.yaml` — гиперпараметры и системный промпт

Этот файл монтируется в контейнер как volume — изменения применяются  
**без пересборки образа**, достаточно `docker compose restart app`.

```yaml
# Параметры генерации LLM
generation:
  max\\\_new\\\_tokens: 300       # Максимальная длина ответа в токенах
  temperature: 0.01         # Близко к 0 = детерминированный ответ
  repetition\\\_penalty: 1.1   # Штраф за повторения

# Системный промпт — инструкция для LLM
system\\\_prompt: |
  Ты — корпоративный ассистент ...

# Параметры поиска
retrieval:
  prefetch\\\_limit: 20        # Кандидатов от dense и sparse поиска
  rerank\\\_candidates: 20     # Передаётся в cross-encoder
  final\\\_top\\\_k: 3            # Финальное кол-во чанков (дефолт для retriever)
  rerank\\\_threshold: -2.0    # Порог отсечения нерелевантных чанков

# Параметры оркестратора
orchestrator:
  top\\\_k: 5                  # Чанков на каждый подвопрос (перекрывает final\\\_top\\\_k)

# Параметры индексации базы знаний
indexing:
  chunk\\\_size: 1000          # Размер чанка в символах
  chunk\\\_overlap: 300        # Перекрытие между чанками
  batch\\\_size: 64            # Батч при загрузке эмбеддингов
  dense\\\_vector\\\_size: 1024   # Размер вектора multilingual-e5-large
```

**Откуда берутся параметры:**

|Параметр|Файл|Источник|
|-|-|-|
|`max\\\_new\\\_tokens`, `temperature`, `repetition\\\_penalty`|`core/generator.py`|`config.yaml`|
|`system\\\_prompt`|`core/generator.py`|`config.yaml`|
|`prefetch\\\_limit`, `rerank\\\_candidates`, `final\\\_top\\\_k`, `rerank\\\_threshold`|`core/retriever.py`|`config.yaml`|
|`top\\\_k` (чанков на подвопрос)|`core/orchestrator.py`|`config.yaml`|
|`chunk\\\_size`, `chunk\\\_overlap`, `batch\\\_size`, `dense\\\_vector\\\_size`|`scripts/index\\\_knowledge\\\_base.py`|`config.yaml`|
|Пути к моделям|все файлы `core/`|`.env`|
|Адреса Qdrant и vLLM|все файлы `core/`|`.env`|

**Примечание про `top\\\_k`:**  
В `retriever.py` есть `final\\\_top\\\_k` — это дефолт когда `search()` вызывается напрямую.  
Оркестратор вызывает `search()` со своим `top\\\_k` из `config.yaml → orchestrator.top\\\_k`,  
который перекрывает `final\\\_top\\\_k`. Фактически работает значение из `orchestrator.top\\\_k`.

\---

## Развёртывание

### Шаг 1 — Подготовка папок

```bash
mkdir -p data/qdrant\\\_storage
```

Убедись что папки с моделями существуют:

```
data/models/multilingual-e5-large/
data/models/bge-reranker-v2-m3/
data/models/qwen2.5-7b-instruct-awq/
```

### Шаг 2 — Настройка `.env`

Открой `.env` и заполни три нижних строки реальными путями с твоей машины:

```env
QDRANT\\\_STORAGE\\\_PATH=/path/to/data/qdrant\\\_storage
MODELS\\\_HOST\\\_PATH=/path/to/data/models
QWEN\\\_MODEL\\\_PATH=/path/to/data/models/qwen2.5-7b-instruct-awq
```

### Шаг 3 — Сборка образа

Выполняется один раз. При первой сборке скачивается PyTorch (\~2GB):

```bash
docker compose build app
```

### Шаг 4 — Запуск Qdrant и vLLM

```bash
docker compose up -d qdrant vllm
```

Подожди пока vLLM загрузит модель (1-3 минуты):

```bash
docker compose logs -f vllm
# Ждёшь строку: "Application startup complete."
```

### Шаг 5 — Индексация базы знаний

Положи файл `knowledge\\\_base\\\_cleaned.json` в папку `data/` и запусти индексацию:

```bash
docker compose run --rm app python scripts/index\\\_knowledge\\\_base.py --recreate
```

Флаг `--recreate` пересоздаёт коллекцию с нуля.  
Без флага — режим upsert, данные дополняются.

Дождись завершения:

```
Готово! Поинтов в коллекции: XXXX
```

**Данные Qdrant сохраняются на диске** в папке `qdrant\\\_storage/`.  
При перезапуске контейнера данные никуда не деваются — Qdrant подхватывает их автоматически.  
Переиндексировать нужно только при изменении базы знаний или параметров чанкинга.

### Шаг 6 — Запуск API-режима (Bitrix, основной режим)

```bash
docker compose up -d
```

По умолчанию поднимутся `qdrant`, `vllm`, `api`.  
Сервис `app` (консольный чат) вынесен в профиль `cli` и **не стартует** командой `docker compose up -d`.

Это важно для VRAM: в Bitrix-режиме ретривер инициализируется только в `api` (один раз), а не дублируется в `app`.

\---

## Использование

### Консольный чат (main.py)

Запуск локально:

```bash
python main.py
```

Запуск через Docker:

```bash
docker compose run --rm -it app python main.py
```

Пример диалога:

```
============================================================
  Корпоративный ассистент Акме
  Введите 'exit' для выхода
============================================================

Вы: Что такое LQA?
Ассистент: LQA (Linguistic Quality Assessment) — оценка лингвистического качества...

Вы: Как оформить отпуск?
Ассистент: Для оформления отпуска необходимо...
  Источники: Инструкция по отпускам, Заявление на отпуск

Вы: exit
Завершение работы.
```

**Важно:** запускай из терминала с поддержкой UTF-8.  
На Windows рекомендуется встроенный терминал PyCharm или ConEmu с UTF-8.  
На Linux/Mac проблем нет.

### Синхронизация с Confluence

#### Ручная загрузка

```bash
# Загрузить страницы из Confluence в JSON
python scripts/sync_confluence.py --download-only

# Загрузить конкретное пространство
python scripts/sync_confluence.py --download-only --spaces TST

# Полный цикл: загрузка + индексация в Qdrant
python scripts/sync_confluence.py

# Только индексация (если JSON уже есть)
python scripts/sync_confluence.py --index-only
```

#### Автоматическая синхронизация

Scheduler запускается как Docker-контейнер и синхронизирует данные по расписанию (по умолчанию: воскресенье 02:00 МСК):

```bash
docker compose up -d scheduler
```

Расписание настраивается в `config.yaml`:

```yaml
confluence_sync:
  schedule:
    day_of_week: 6  # 0=понедельник, 6=воскресенье
    hour: 2
    minute: 0
```

\---

## Обновление параметров

### Изменить системный промпт или гиперпараметры

1. Отредактируй `config.yaml`
2. Перезапусти app — пересборка не нужна:

```bash
docker compose restart app
```

### Изменить код (`.py` файлы)

Пересборка образа:

```bash
docker compose build app
docker compose up -d
```

Быстро — слой с зависимостями закеширован, пересобирается только слой с кодом.

### Обновить зависимости (`requirements.txt`)

```bash
docker compose build app  # пересоберёт слой с pip install
docker compose up -d
```

\---

## Полезные команды

```bash
# Запустить основной Bitrix-режим (без app)
docker compose up -d

# Остановить (данные Qdrant сохраняются)
docker compose down

# Логи конкретного сервиса
docker compose logs -f api
docker compose logs -f vllm
docker compose logs -f qdrant
docker compose logs -f scheduler

# Перезапустить api после изменения config.yaml
docker compose restart api

# Пересобрать api после изменения кода
docker compose build api && docker compose up -d

# Загрузить данные из Confluence и проиндексировать
python scripts/sync_confluence.py --download-only
python scripts/sync_confluence.py --index-only

# Переиндексировать базу знаний
docker compose run --rm app python scripts/sync_confluence.py --index-only

# Запустить scheduler для автоматической синхронизации
docker compose up -d scheduler

# Проверить что Qdrant жив и данные на месте
curl http://localhost:6333/collections/acme_knowledge

# Проверить что vLLM жив
curl http://localhost:8000/health
```

---

## Сетевая схема

```
[Bitrix24]              [Confluence]
     │                       │
     ▼                       ▼
  [api: rag-api]     [scheduler]
   core/orchestrator.py  sync_confluence.py
     │                       │
     └───────────┬───────────┘
                 │
           ┌─────┴─────┐
           │           │
           ▼           ▼
     [qdrant:6333] [vllm:8000]
      Векторная БД  Qwen 2.5-7B
      (данные на    (модель на
        диске)        GPU)
```

Все сервисы в одной Docker-сети.  
Внутри сети общаются по имени сервиса: `qdrant`, `vllm`.  
С хоста доступны через `localhost:6333` и `localhost:8000`.

\---

## Устранение проблем

**vLLM не запускается**

```bash
docker compose logs vllm
# Убедись что GPU доступна: nvidia-smi
# Убедись что путь к модели верный в .env → QWEN\\\_MODEL\\\_PATH
```

**Qdrant не запускается**

```bash
docker compose logs qdrant
# Убедись что папка существует: ls data/qdrant\\\_storage
# Убедись что путь верный в .env → QDRANT\\\_STORAGE\\\_PATH
```

**Модели не найдены (retriever)**

```bash
# Убедись что папка с моделями смонтирована
docker compose exec app ls /app/models
# Должны быть: multilingual-e5-large/ bge-reranker-v2-m3/
```

**Битые символы при вводе кириллицы**  
Запускай из терминала с поддержкой UTF-8 (PyCharm, ConEmu).  
На сервере с Linux проблемы нет.

---

## Синхронизация с Confluence

### Автоматический режим (по расписанию)

Синхронизация выполняется раз в неделю в воскресенье в 02:00 по МСК.

```bash
# Запуск планировщика
docker compose up -d scheduler
```

Расписание настраивается в `config.yaml`:
```yaml
confluence_sync:
  schedule:
    day_of_week: 6    # 0=Пн ... 6=Вс
    hour: 2           # час (MSK)
    minute: 0
```

### Ручной режим

```bash
# Полная синхронизация: загрузка из Confluence + индексация в Qdrant
docker compose run --rm app python scripts/sync_confluence.py

# Только загрузка из Confluence в JSON (без индексации)
docker compose run --rm app python scripts/sync_confluence.py --download-only

# Только индексация из существующего JSON
# 1. Положи файл data/knowledge_base.json
docker compose run --rm app python scripts/sync_confluence.py --index-only
```

### Конфигурация Confluence

Добавь в `.env`:
```env
CONFLUENCE_URL=https://confluence.example.com
CONFLUENCE_USERNAME=your_username
CONFLUENCE_PASSWORD=your_password_or_token
CONFLUENCE_SPACES=SPACE1,SPACE2   # пусто = все пространства
```

### Формат данных

JSON-файл `data/knowledge_base.json` имеет ту же структуру, что и `knowledge_base_cleaned.json`:
```json
{
  "meta": { "exported_at": "...", "sources": ["confluence"], "total_docs": 959 },
  "documents": [
    {
      "id": "confluence_12345",
      "title": "Название страницы",
      "breadcrumbs": ["Space", "Раздел", "Подраздел"],
      "sections": [{ "heading": "Заголовок", "text": "Текст секции" }],
      ...
    }
  ]
}
```

### Снапшоты Qdrant

Перед индексацией автоматически создаётся снапшот текущей коллекции.  
Снапшоты хранятся в `data/qdrant_snapshots/`.

### Жизненный цикл

1. Загрузка страниц из Confluence API
2. Конвертация в JSON-формат knowledge_base
3. Создание снапшота текущей базы Qdrant
4. Индексация новой базы в Qdrant (с пересозданием коллекции)
5. Удаление JSON-файла с сервера
