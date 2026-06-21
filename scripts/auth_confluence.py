"""
scripts/auth_confluence.py
==========================
Скрипт первичной авторизации в Confluence через OAuth 2.0.

Запуск:
  python scripts/auth_confluence.py

Процесс:
  1. Генерирует URL для Consent Screen
  2. Открывает браузер
  3. Получает authorization code от пользователя
  4. Обменивает code на tokens
  5. Сохраняет tokens в .confluence_tokens.json
"""

import logging
import os
import sys
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from confluence_auth import ConfluenceOAuth2


def main():
    client_id = os.getenv("CONFLUENCE_CLIENT_ID", "")
    client_secret = os.getenv("CONFLUENCE_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        log.error("CONFLUENCE_CLIENT_ID и CONFLUENCE_CLIENT_SECRET должны быть заданы в .env")
        sys.exit(1)

    oauth = ConfluenceOAuth2()

    print("\n" + "=" * 60)
    print("Авторизация в Confluence через OAuth 2.0")
    print("=" * 60)

    auth_url = oauth.get_authorization_url()

    print("\n1. Откройте эту ссылку в браузере:")
    print("-" * 60)
    print(auth_url)
    print("-" * 60)

    try:
        webbrowser.open(auth_url)
        print("\nБраузер открыт автоматически.")
    except Exception:
        print("\nНе удалось открыть браузер автоматически.")

    print("\n2. Войдите в Confluence и нажмите 'Allow'")
    print("3. Скопируйте authorization code из URL")
    print("   (URL будет выглядеть как: redirect_uri?code=XXXXXX)\n")

    code = input("Вставьте authorization code: ").strip()

    if not code:
        log.error("Authorization code не может быть пустым")
        sys.exit(1)

    print("\nОбмен code на tokens...")
    try:
        token_data = oauth.exchange_code_for_token(code)
        print("\n" + "=" * 60)
        print("Авторизация успешна!")
        print("=" * 60)
        print(f"Access token:  {token_data['access_token'][:30]}...")
        print(f"Expires in:    {token_data.get('expires_in')} seconds")
        print(f"Refresh token: {'получен' if token_data.get('refresh_token') else 'не получен'}")
        print("=" * 60)

        print("\nПроверка подключения...")
        if oauth.test_connection():
            print("\nПодключение к Confluence работает!")
        else:
            print("\nПредупреждение: подключение не проверено")

    except Exception as e:
        log.error(f"Ошибка при обмене code на tokens: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
