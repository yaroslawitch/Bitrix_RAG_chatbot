"""
scripts/confluence_auth.py
==========================
OAuth 2.0 аутентификация для Confluence Data Center.

Использует Authorization Code flow для получения access_token.

Переменные окружения (из .env):
  CONFLUENCE_CLIENT_ID     — ID приложения
  CONFLUENCE_CLIENT_SECRET — Секрет приложения
  CONFLUENCE_REDIRECT_URI  — Callback URL
  CONFLUENCE_ACCESS_TOKEN  — Access token (заполняется автоматически)
  CONFLUENCE_REFRESH_TOKEN — Refresh token (заполняется автоматически)
"""

import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

# ── Конфигурация ─────────────────────────────────────────────────────────────

CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "")
CLIENT_ID = os.getenv("CONFLUENCE_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CONFLUENCE_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("CONFLUENCE_REDIRECT_URI", "http://localhost:8080/confluence/callback")

# Путь к файлу для хранения токенов
TOKEN_FILE = Path(__file__).parent.parent / ".confluence_tokens.json"


class ConfluenceOAuth2:
    """OAuth 2.0 клиент для Confluence Data Center."""

    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        base_url: str = None,
        redirect_uri: str = None,
    ):
        self.client_id = client_id or CLIENT_ID
        self.client_secret = client_secret or CLIENT_SECRET
        self.base_url = (base_url or CONFLUENCE_URL).rstrip("/")
        self.redirect_uri = redirect_uri or REDIRECT_URI

        self.access_token = None
        self.refresh_token = None
        self.token_expiry = 0

        self._load_tokens()

    def get_authorization_url(self, scopes: list[str] = None) -> str:
        """
        Генерирует URL для Consent Screen.

        Args:
            scopes: Список scopes (по умолчанию READ)

        Returns:
            URL для открытия в браузере
        """
        if scopes is None:
            scopes = ["READ"]

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
        }

        auth_url = f"{self.base_url}/rest/oauth2/latest/authorize?{urlencode(params)}"
        log.info(f"[OAuth] Authorization URL: {auth_url}")
        return auth_url

    def exchange_code_for_token(self, code: str) -> dict:
        """
        Обменивает authorization code на access_token.

        Args:
            code: Authorization code, полученный от Confluence

        Returns:
            dict с токенами
        """
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri,
        }

        token_url = f"{self.base_url}/rest/oauth2/latest/token"
        resp = requests.post(token_url, data=data, timeout=30)
        resp.raise_for_status()

        token_data = resp.json()
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data.get("refresh_token")
        self.token_expiry = time.time() + token_data.get("expires_in", 7200)

        self._save_tokens()

        log.info("[OAuth] Tokens получены успешно")
        log.info(f"[OAuth] Access token: {self.access_token[:20]}...")
        log.info(f"[OAuth] Expires in: {token_data.get('expires_in')} seconds")

        return token_data

    def refresh_access_token(self) -> dict:
        """
        Обновляет access_token через refresh_token.

        Returns:
            dict с новыми токенами
        """
        if not self.refresh_token:
            raise ValueError("Refresh token отсутствует. Необходима повторная авторизация.")

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
            "redirect_uri": self.redirect_uri,
        }

        token_url = f"{self.base_url}/rest/oauth2/latest/token"
        resp = requests.post(token_url, data=data, timeout=30)
        resp.raise_for_status()

        token_data = resp.json()
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data.get("refresh_token", self.refresh_token)
        self.token_expiry = time.time() + token_data.get("expires_in", 7200)

        self._save_tokens()

        log.info("[OAuth] Access token обновлён")
        return token_data

    def get_valid_token(self) -> str:
        """
        Получает валидный access_token (с автоматическим обновлением).

        Returns:
            Access token string
        """
        if not self.access_token:
            raise ValueError(
                "Access token отсутствует. "
                "Запустите scripts/auth_confluence.py для авторизации."
            )

        if time.time() >= self.token_expiry - 60:
            log.info("[OAuth] Token истекает, обновляем...")
            self.refresh_access_token()

        return self.access_token

    def _save_tokens(self):
        """Сохраняет токены в файл."""
        token_data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_expiry": self.token_expiry,
        }
        with open(TOKEN_FILE, "w") as f:
            json.dump(token_data, f, indent=2)
        log.debug("[OAuth] Tokens сохранены в %s", TOKEN_FILE)

    def _load_tokens(self):
        """Загружает токены из файла."""
        if TOKEN_FILE.exists():
            try:
                with open(TOKEN_FILE) as f:
                    token_data = json.load(f)
                self.access_token = token_data.get("access_token")
                self.refresh_token = token_data.get("refresh_token")
                self.token_expiry = token_data.get("token_expiry", 0)
                log.debug("[OAuth] Tokens загружены из %s", TOKEN_FILE)
            except (json.JSONDecodeError, KeyError):
                log.warning("[OAuth] Не удалось загрузить tokens из файла")

    def test_connection(self) -> bool:
        """Проверяет подключение к Confluence с текущим токеном."""
        try:
            token = self.get_valid_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            }
            resp = requests.get(
                f"{self.base_url}/rest/api/content",
                headers=headers,
                params={"limit": 1},
                timeout=30,
            )
            resp.raise_for_status()
            log.info("[OAuth] Подключение успешно")
            return True
        except Exception as e:
            log.error(f"[OAuth] Ошибка подключения: {e}")
            return False


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    oauth = ConfluenceOAuth2()

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        oauth.test_connection()
    else:
        auth_url = oauth.get_authorization_url()
        print("\n" + "=" * 60)
        print("Откройте эту ссылку в браузере для авторизации:")
        print("=" * 60)
        print(auth_url)
        print("=" * 60)
        print("\nПосле авторизации скопируйте authorization code из URL")
        print("и вставьте его сюда.\n")
