"""Tool registry: tool descriptions come from prompts/tool_descriptions.txt.

The registry builds (1) a neutral list of ToolSpec objects that the LLM wrapper turns into
Gemini function declarations, and (2) a name -> function dictionary used to run the calls.
"""
import logging
import re
from dataclasses import dataclass
from typing import Callable

from config import PROMPTS_DIR
from tools.calculator import calculator
from tools.document_qa import document_qa
from tools.tavily_search import tavily_search

logger = logging.getLogger(__name__)

TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "document_qa": document_qa,
    "tavily_search": tavily_search,
    "calculator": calculator,
}


@dataclass
class ToolSpec:
    name: str
    description: str
    param: str            # every tool takes one string parameter


def load_tool_specs() -> list[ToolSpec]:
    """Read '[name]' sections (a 'param: x' line, then the description) from the prompts folder."""
    text = (PROMPTS_DIR / "tool_descriptions.txt").read_text(encoding="utf-8")
    specs = []
    for name, body in re.findall(r"^\[(\w+)\]\n(.*?)(?=^\[\w+\]\n|\Z)", text, flags=re.M | re.S):
        param_line, _, description = body.strip().partition("\n")
        specs.append(ToolSpec(name, description.strip(), param_line.split(":", 1)[1].strip()))
    return specs


def run_tool(name: str, args: dict) -> str:
    """Run one tool call. Never raises, so the agent loop always continues."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        return func(**args)
    except Exception as exc:                       # tools handle expected errors themselves
        logger.exception("Tool %s failed", name)
        return f"ERROR: {name} failed: {exc}"
