"""Make the src/ folder importable in tests (the app imports 'utils', 'tools', 'config')."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
