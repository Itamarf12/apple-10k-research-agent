"""Tavily web search tool: current market data, competitor info and news."""
import logging
import os
from datetime import date

logger = logging.getLogger(__name__)


def tavily_search(query: str) -> str:
    """Search the web for current information. Returns up to 5 results with URL and date."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "ERROR: web search unavailable (TAVILY_API_KEY is not set)"
    try:
        from tavily import TavilyClient
        response = TavilyClient(api_key=api_key).search(query=query, max_results=5)
    except Exception as exc:                       # network, quota, invalid key ...
        logger.warning("Tavily search failed: %s", exc)
        return f"ERROR: web search unavailable ({exc})"
    results = response.get("results", [])
    if not results:
        return "NO_RESULTS: the web search returned nothing for this query."
    today = date.today().isoformat()
    return "\n\n".join(
        f"[Tavily: {r.get('url', '')} | {r.get('title', 'untitled')} | retrieved {today}]\n{r.get('content', '')}"
        for r in results)
