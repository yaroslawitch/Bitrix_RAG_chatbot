"""
api/bitrix_webhook.py
=====================
FastAPI-сервер для интеграции с Bitrix24 Chatbot API (imbot, v1).

Схема работы:
  1. Bitrix24 шлёт POST на /webhook при каждом сообщении боту
  2. Сервер извлекает текст, передаёт в Orchestrator
  3. Ответ отправляется обратно в Bitrix24 через imbot.message.add

Формат входящего события ONIMBOTMESSAGEADD (из документации Bitrix24):
  {
    "event": "ONIMBOTMESSAGEADD",
    "data": {
      "BOT": { "567": { "access_token": "...", "BOT_ID": "567", ... } },
      "PARAMS": { "MESSAGE": "текст", "DIALOG_ID": "27", "TO_USER_ID": "567", ... },
      "USER": { "ID": "27", "NAME": "...", ... }
    },
    "auth": { "access_token": "...", "application_token": "..." }
  }

Формат ответа через imbot.message.add:
  POST {BITRIX_WEBHOOK_URL}/imbot.message.add?auth={BOT_TOKEN}
  Body: { "BOT_ID": 567, "DIALOG_ID": "27", "MESSAGE": "текст ответа" }

Переменные окружения (из .env):
  BITRIX_WEBHOOK_URL   — URL вида https://your.bitrix24.ru/rest/USER_ID/TOKEN/
  BITRIX_BOT_TOKEN     — токен бота (для auth-параметра)
  BITRIX_CLIENT_ID     — CLIENT_ID приложения (обязателен для webhook-вызовов)
  WEBHOOK_SECRET       — опциональный токен для валидации входящих запросов
  APP_HOST             — хост (default: 0.0.0.0)
  APP_PORT             — порт (default: 8080)
  EVENT_HANDLER_URL    — публичный URL webhook
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import partial

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Настройки Bitrix24 (из .env) ──────────────────────────────────────────────
BITRIX_WEBHOOK_URL = os.getenv("BITRIX_WEBHOOK_URL", "")   # https://your.bitrix24.ru/rest/1/token/
BITRIX_BOT_TOKEN   = os.getenv("BITRIX_BOT_TOKEN",   "")   # botToken бота
BITRIX_CLIENT_ID   = os.getenv("BITRIX_CLIENT_ID",   "")   # CLIENT_ID приложения
WEBHOOK_SECRET     = os.getenv("WEBHOOK_SECRET",      "")   # опциональная валидация
APP_HOST           = os.getenv("APP_HOST",            "0.0.0.0")
APP_PORT           = int(os.getenv("APP_PORT",        "8080"))
EVENT_HANDLER_URL  = os.getenv("EVENT_HANDLER_URL", "")

# ── Глобальный оркестратор ─────────────────────────────────────────────────────
orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализируем оркестратор один раз при старте сервера."""
    global orchestrator
    log.info("[API] Инициализация оркестратора...")
    from core.orchestrator import Orchestrator
    orchestrator = Orchestrator()
    log.info("[API] Оркестратор готов. Сервер запущен.")
    yield
    log.info("[API] Сервер остановлен.")


app = FastAPI(lifespan=lifespan)


# ── Извлечение данных из входящего события Bitrix24 ────────────────────────────

def _extract_event_data(data: dict) -> dict:
    """
    Извлекает bot_id, text, dialog_id из входящего JSON-события Bitrix24.

    Поддерживает два формата:
      1. Стандартный Bitrix24 v1 (ONIMBOTMESSAGEADD):
         data["PARAMS"]["MESSAGE"], data["PARAMS"]["DIALOG_ID"], data["BOT"]
      2. Упрощённый формат (для mock-тестирования):
         data["MESSAGE"], data["DIALOG_ID"], data["BOT_ID"]
         или data["message"]["text"], data["chat"]["dialogId"], data["bot"]["id"]
    """
    event_data = data.get("data", {})

    # ── Стандартный формат Bitrix24 v1 (ONIMBOTMESSAGEADD) ──
    # Текст: data.PARAMS.MESSAGE
    params = event_data.get("PARAMS", {})
    text = params.get("MESSAGE", "")

    # DIALOG_ID: data.PARAMS.DIALOG_ID
    dialog_id = params.get("DIALOG_ID", "")

    # BOT_ID: data.BOT — это словарь {BOT_ID: {auth_data}}, ключ — ID бота
    bot_id = 0
    bots = event_data.get("BOT", {})
    if bots and isinstance(bots, dict):
        # Берём первого (и обычно единственного) бота
        for key, bot_info in bots.items():
            bot_id = int(key)
            break

    # Если BOT_ID не найден через data.BOT, пробуем TO_USER_ID из PARAMS
    if not bot_id:
        bot_id = int(params.get("TO_USER_ID", 0) or 0)

    # ── Fallback: упрощённый формат (mock / альтернативные версии) ──
    if not text:
        text = event_data.get("MESSAGE", "")
    if not text:
        text = event_data.get("message", {}).get("text", "")

    if not dialog_id:
        dialog_id = event_data.get("DIALOG_ID", "")
    if not dialog_id:
        dialog_id = event_data.get("chat", {}).get("dialogId", "")

    if not bot_id:
        bot_id = int(event_data.get("BOT_ID", 0) or 0)
    if not bot_id:
        bot_id = int(event_data.get("bot", {}).get("id", 0) or 0)

    return {
        "bot_id":    bot_id,
        "text":      text.strip() if text else "",
        "dialog_id": dialog_id,
    }


# ── Отправка ответа в Bitrix24 ─────────────────────────────────────────────────

async def send_message_to_bitrix(dialog_id: str, bot_id: int, text: str):
    """
    Отправляет сообщение в чат Bitrix24 через imbot.message.add (v1 API).

    Формат запроса (из документации):
      POST {BITRIX_WEBHOOK_URL}/imbot.message.add?auth={BOT_TOKEN}
      {
        "BOT_ID": 39,
        "DIALOG_ID": "chat123",
        "MESSAGE": "Текст сообщения"
      }

    Если используется webhook-авторизация, передаётся CLIENT_ID.
    Если OAuth — передаётся auth с access_token.
    """
    if not BITRIX_WEBHOOK_URL:
        log.error("[API] BITRIX_WEBHOOK_URL не задан в .env")
        return

    url = BITRIX_WEBHOOK_URL.rstrip("/") + "/imbot.message.add"

    # Авторизация: webhook (auth=token в query) или OAuth (auth в body)
    params = {}
    if BITRIX_BOT_TOKEN:
        params["auth"] = BITRIX_BOT_TOKEN

    payload = {
        "BOT_ID"   : bot_id,
        "DIALOG_ID": dialog_id,
        "MESSAGE"  : text,
    }

    # Для webhook-методов обязателен CLIENT_ID
    if BITRIX_CLIENT_ID:
        payload["CLIENT_ID"] = BITRIX_CLIENT_ID

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, params=params)
            if resp.status_code != 200:
                log.warning(
                    f"[API] Bitrix24 вернул статус {resp.status_code}: "
                    f"{resp.text[:300]}"
                )
            else:
                log.info(f"[API] Сообщение отправлено в Bitrix24. dialog_id={dialog_id}")
    except Exception as e:
        log.error(f"[API] Ошибка отправки в Bitrix24: {e}")


# ── Обработка событий (синхронная, для запуска в executor) ─────────────────────

def _process_message_sync(bot_id: int, text: str, dialog_id: str):
    """
    Синхронная обработка сообщения через RAG-пайплайн.
    Запускается в thread pool, чтобы не блокировать event loop.
    """
    try:
        result = orchestrator.answer(text)
        return result.answer
    except Exception as e:
        log.error(f"[API] Ошибка оркестратора: {e}")
        return "Произошла ошибка при обработке запроса. Попробуйте повторить позже."


# ── Webhook endpoint ───────────────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(request: Request, x_bitrix_signature: str | None = Header(default=None)):
    """
    Принимает события от Bitrix24.

    Обрабатываемые события:
      ONIMBOTMESSAGEADD    — новое сообщение боту
      ONIMBOTJOINCHAT      — бот добавлен в чат
      ONAPPINSTALL         — установка приложения

    Важно: Bitrix24 ожидает ответ за ~5 секунд. Долгая обработка (RAG-пайплайн
    может занимать до 60+ сек) приведёт к повторной отправке события.
    Поэтому мы:
      1. Быстро парсим входящий запрос
      2. Возвращаем 200 OK
      3. Обрабатываем сообщение асинхронно в фоне
    """
    try:
        data = await request.json()
    except Exception:
        log.warning("[API] Получен невалидный JSON")
        return JSONResponse(content={"result": False}, status_code=400)

    event = data.get("event", "")

    log.info(f"[API] Событие: {event}")

    # ── Валидация токена (опционально) ────────────────────────────────────────
    if WEBHOOK_SECRET:
        auth = data.get("auth", {})
        incoming_token = auth.get("application_token", "")
        header_token = x_bitrix_signature or ""
        if incoming_token != WEBHOOK_SECRET and header_token != WEBHOOK_SECRET:
            log.warning("[API] Неверный webhook secret — запрос отклонён")
            return JSONResponse(
                content={"result": False, "error": "Invalid token"},
                status_code=401,
            )

    # ── Обработка нового сообщения ────────────────────────────────────────────
    if event == "ONIMBOTMESSAGEADD":
        extracted = _extract_event_data(data)
        bot_id    = extracted["bot_id"]
        text      = extracted["text"]
        dialog_id = extracted["dialog_id"]

        if not text:
            log.warning("[API] Пустое сообщение — пропускаем")
            return JSONResponse(content={"result": True})

        log.info(f"[API] Вопрос от пользователя: '{text[:80]}'")

        # Запускаем обработку в фоне, чтобы вернуть 200 OK сразу
        # run_in_executor запускает синхронный код в thread pool, не блокируя event loop
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            None,
            _process_and_send,
            bot_id,
            text,
            dialog_id,
        )

    # ── Бот добавлен в чат ────────────────────────────────────────────────────
    elif event == "ONIMBOTJOINCHAT":
        extracted = _extract_event_data(data)
        bot_id    = extracted["bot_id"]
        dialog_id = extracted["dialog_id"]

        welcome = "Привет! Я корпоративный ассистент Акме. Задайте мне вопрос по базе знаний."
        await send_message_to_bitrix(dialog_id, bot_id, welcome)

    # ── Установка приложения ──────────────────────────────────────────────────
    elif event == "ONAPPINSTALL":
        log.info("[API] Приложение установлено в Bitrix24")

    return JSONResponse(content={"result": True})


def _process_and_send(bot_id: int, text: str, dialog_id: str):
    """Обрабатывает сообщение и отправляет ответ (вызывается в executor)."""
    answer = _process_message_sync(bot_id, text, dialog_id)

    # Отправляем ответ синхронно через httpx (мы уже в executor, не блокируем event loop)
    if BITRIX_BOT_TOKEN or BITRIX_CLIENT_ID:
        _send_message_sync(dialog_id, bot_id, answer)
    else:
        log.error("[API] Ни BITRIX_BOT_TOKEN, ни BITRIX_CLIENT_ID не заданы")


def _send_message_sync(dialog_id: str, bot_id: int, text: str):
    """Синхронная отправка сообщения в Bitrix24 (вызывается в executor)."""
    if not BITRIX_WEBHOOK_URL:
        log.error("[API] BITRIX_WEBHOOK_URL не задан в .env")
        return

    url = BITRIX_WEBHOOK_URL.rstrip("/") + "/imbot.message.add"

    params = {}
    if BITRIX_BOT_TOKEN:
        params["auth"] = BITRIX_BOT_TOKEN

    payload = {
        "BOT_ID"   : bot_id,
        "DIALOG_ID": dialog_id,
        "MESSAGE"  : text,
    }

    if BITRIX_CLIENT_ID:
        payload["CLIENT_ID"] = BITRIX_CLIENT_ID

    try:
        import requests
        resp = requests.post(url, json=payload, params=params, timeout=30)
        if resp.status_code != 200:
            log.warning(
                f"[API] Bitrix24 вернул статус {resp.status_code}: "
                f"{resp.text[:300]}"
            )
        else:
            log.info(f"[API] Сообщение отправлено в Bitrix24. dialog_id={dialog_id}")
    except Exception as e:
        log.error(f"[API] Ошибка отправки в Bitrix24: {e}")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "orchestrator": orchestrator is not None,
        "event_handler": EVENT_HANDLER_URL or f"http://{APP_HOST}:{APP_PORT}/webhook",
    }


# ── Запуск ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.bitrix_webhook:app", host=APP_HOST, port=APP_PORT, reload=False)
