"""Logging: everything goes to logs/agent.log; only warnings reach the terminal."""
import logging

from config import LOG_DIR


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    handler = logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.CRITICAL)      # details go to logs/agent.log
    logging.basicConfig(level=logging.INFO, handlers=[handler, console], force=True)
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
