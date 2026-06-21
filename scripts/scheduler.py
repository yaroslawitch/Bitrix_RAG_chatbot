"""
scripts/scheduler.py
====================
Планировщик задач для автоматической синхронизации с Confluence.

Запуск:
    python scripts/scheduler.py

По умолчанию выполняет sync_confluence.py раз в неделю:
  - Воскресенье в 02:00 по Moscow time (UTC+3)

Настройка расписания — в config.yaml (раздел confluence_sync.schedule).
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

MOSCOW_TZ = timezone(timedelta(hours=3))


def _load_config() -> dict:
    config_path = Path(os.getenv("CONFIG_PATH", str(Path(__file__).parent.parent / "config.yaml")))
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_next_run_time(config: dict) -> datetime:
    """
    Вычисляет время следующего запуска по расписанию.
    Формат расписания в config.yaml:
      confluence_sync:
        schedule:
          day_of_week: 0   # 0=воскресенье, 1=понедельник, ...
          hour: 2
          minute: 0
    """
    schedule = config.get("confluence_sync", {}).get("schedule", {})
    target_dow = schedule.get("day_of_week", 6)  # по умолчанию воскресенье
    target_hour = schedule.get("hour", 2)
    target_minute = schedule.get("minute", 0)

    now = datetime.now(MOSCOW_TZ)
    days_ahead = (target_dow - now.weekday()) % 7

    if days_ahead == 0:
        target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        if target_time <= now:
            days_ahead = 7

    next_run = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    next_run += timedelta(days=days_ahead)

    return next_run


def run_sync():
    """Запускает sync_confluence.py."""
    log.info("[Scheduler] Запуск синхронизации Confluence...")

    script_path = Path(__file__).parent / "sync_confluence.py"
    exit_code = os.system(f'"{sys.executable}" "{script_path}"')

    if exit_code == 0:
        log.info("[Scheduler] Синхронизация завершена успешно")
    else:
        log.error(f"[Scheduler] Синх завершена с ошибкой (код: {exit_code})")


def main():
    config = _load_config()
    schedule = config.get("confluence_sync", {}).get("schedule", {})

    day_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    target_dow = schedule.get("day_of_week", 6)
    target_hour = schedule.get("hour", 2)
    target_minute = schedule.get("minute", 0)

    log.info("=" * 60)
    log.info("[Scheduler] Планировщик синхронизации Confluence")
    log.info(f"[Scheduler] Расписание: {day_names[target_dow]} {target_hour:02d}:{target_minute:02d} (MSK)")
    log.info("[Scheduler] Нажмите Ctrl+C для остановки")
    log.info("=" * 60)

    while True:
        next_run = get_next_run_time(config)
        now = datetime.now(MOSCOW_TZ)
        wait_seconds = (next_run - now).total_seconds()

        log.info(f"[Scheduler] Следующий запуск: {next_run.strftime('%Y-%m-%d %H:%M:%S')} MSK")
        log.info(f"[Scheduler] Ожидание: {wait_seconds/3600:.1f} часов")

        try:
            time.sleep(wait_seconds)
        except KeyboardInterrupt:
            log.info("[Scheduler] Остановлен пользователем")
            break

        run_sync()

        config = _load_config()


if __name__ == "__main__":
    main()
