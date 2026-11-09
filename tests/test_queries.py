"""The two required test queries. They call the real Gemini and Tavily APIs.

Run: pytest tests/test_queries.py -s      (needs .env keys and the built index)
"""
import os

import pytest
from dotenv import load_dotenv

from config import CHROMA_DIR, ROOT

load_dotenv(ROOT / ".env")
pytestmark = pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("TAVILY_API_KEY") and CHROMA_DIR.exists()),
    reason="needs GEMINI_API_KEY, TAVILY_API_KEY and the index (python src/build_index.py)")

QUERY_1 = ("What are Apple's top 3 risk factors mentioned in their latest 10-K, "
           "and what percentage of total revenue did they spend on R&D?")
QUERY_2 = ("How does Apple's gross margin compare to Microsoft's current gross margin, "
           "and what reasons does Apple cite in their 10-K for any margin pressure?")


@pytest.fixture(scope="module")
def agent():
    from agent import Agent
    from utils.logging_setup import setup_logging
    setup_logging()
    return Agent()


def test_query_1_document_only(agent):
    answer, trace = agent.run_with_trace(QUERY_1)
    names = [t.name for t in trace]
    print("\nTOOLS:", names, "\n", answer)
    assert "document_qa" in names
    assert "tavily_search" not in names              # everything is in the 10-K
    assert "calculator" in names
    assert "8.3" in answer                           # 34,550 / 416,161 = 8.30%
    assert "Item 1A" in answer or "Risk Factors" in answer


def test_query_2_hybrid(agent):
    answer, trace = agent.run_with_trace(QUERY_2)
    names = [t.name for t in trace]
    print("\nTOOLS:", names, "\n", answer)
    assert "document_qa" in names and "tavily_search" in names
    assert "46.9" in answer                          # Apple's FY2025 gross margin percentage
    assert "Microsoft" in answer
    assert "tariff" in answer.lower()                # reason cited in the MD&A
