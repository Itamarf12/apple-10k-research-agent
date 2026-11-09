"""Central settings: paths and tuning constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HTML_PATH = DATA_DIR / "apple_10k_2025.htm"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))   # override if the disk does not support SQLite
TABLES_PATH = DATA_DIR / "tables.json"
PROMPTS_DIR = ROOT / "prompts"
LOG_DIR = ROOT / "logs"

COLLECTION = "apple_10k"
DOC_LABEL = "10-K FY2025"        # shown in every 10-K citation
DEFAULT_MODEL = "gemini-3.8-flash"

CHUNK_WORDS = 170                # target chunk size; MiniLM embeds at most 256 tokens (about 190 words)
TOP_K = 5                        # chunks returned by the vector search
MAX_CHUNKS = 10                  # cap after following links and cross-references
MAX_DISTANCE = 0.9               # cosine distance above which a hit counts as irrelevant
MAX_STEPS = 8                    # agent loop limit
