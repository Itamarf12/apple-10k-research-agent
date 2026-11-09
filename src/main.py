"""Command-line entry point.

Usage:
  python src/main.py                 interactive prompt
  python src/main.py -q "question"   answer one question and exit
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so "utils", "tools", "config" import

from dotenv import load_dotenv

from config import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Apple 10-K research agent")
    parser.add_argument("-q", "--question", help="answer one question and exit")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    missing = [k for k in ("GEMINI_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}. Copy .env.example to .env and add your keys.")

    from agent import Agent                      # imported after the environment is loaded
    from utils.logging_setup import setup_logging
    from utils.store import Store

    setup_logging()
    try:
        Store()                                  # fail early if the index has not been built
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    agent = Agent()

    if args.question:
        print(agent.run(args.question))
        return
    print("Apple 10-K agent. Type a question, or 'exit'.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit"):
            break
        if question:
            print("[Agent processing...]")
            print(agent.run(question))
            print()


if __name__ == "__main__":
    main()
