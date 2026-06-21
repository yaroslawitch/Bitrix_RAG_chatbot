"""
main.py — консольный интерфейс для RAG-пайплайна RAG-Chatbot
==============================================================
Запуск:
    python main.py

Введи вопрос и нажми Enter. Введи 'exit' или 'quit' для выхода.
"""

import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

from core.orchestrator import Orchestrator


def main():
    print("\n" + "="*60)
    print("  Корпоративный ассистент Акме")
    print("  Введите 'exit' для выхода")
    print("="*60 + "\n")

    log.info("Инициализация компонентов...")
    orch = Orchestrator()
    log.info("Готов к работе.\n")

    while True:
        try:
            question = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение работы.")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "выход"):
            print("Завершение работы.")
            break

        result = orch.answer(question)

        print(f"\nАссистент:\n{result.answer}")

        if result.sub_answers:
            sources = []
            for sa in result.sub_answers:
                for s in sa.sources:
                    title = s.get("title", "")
                    if title and title not in sources:
                        sources.append(title)
            if sources:
                print(f"\nИсточники: {', '.join(sources)}")

        print()


if __name__ == "__main__":
    main()
