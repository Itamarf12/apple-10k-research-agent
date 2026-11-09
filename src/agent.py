"""Agent: a planning step, then a function-calling loop (reason -> call tools -> read results).

Gemini decides which tools to call and in what order. Our code only runs the calls it asks
for and feeds the results back, until Gemini answers without asking for a tool.
"""
import logging
from dataclasses import dataclass
from datetime import date

from config import MAX_STEPS, PROMPTS_DIR
from tools.registry import load_tool_specs, run_tool
from utils.llm import GeminiLLM, LLMError

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    args: dict


def _read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


class Agent:
    def __init__(self, llm=None, verbose: bool = True) -> None:
        self.verbose = verbose
        system = _read_prompt("system_prompt.txt").replace("{today}", date.today().isoformat())
        examples = PROMPTS_DIR / "examples.txt"
        if examples.exists():
            system += "\n\n" + examples.read_text(encoding="utf-8")
        self.llm = llm or GeminiLLM(system, load_tool_specs())
        self.plan_prompt = _read_prompt("plan_prompt.txt")

    def _say(self, message: str) -> None:
        if self.verbose:
            print(message)

    def run(self, question: str) -> str:
        return self.run_with_trace(question)[0]

    def run_with_trace(self, question: str) -> tuple[str, list[ToolCall]]:
        """Answer a question. Returns the answer and the list of tool calls that were made."""
        trace: list[ToolCall] = []
        try:
            plan = self.llm.plan(self.plan_prompt.replace("{question}", question))
            self._say(f"[Plan]\n{plan}")
            messages = [self.llm.user_message(f"{question}\n\nYour plan:\n{plan}")]

            for step in range(MAX_STEPS):
                reply = self.llm.generate(messages)
                if reply.content is not None:
                    messages.append(reply.content)
                if not reply.function_calls:                  # no tool requested: final answer
                    return reply.text, trace
                results = []
                for call in reply.function_calls:             # Gemini may ask for several at once
                    self._say(f"[Tool: {call.name}] {call.args}")
                    logger.info("Step %d: %s %s", step + 1, call.name, call.args)
                    trace.append(ToolCall(call.name, call.args))
                    output = run_tool(call.name, call.args)
                    if output.startswith(("ERROR", "NO_RESULTS")):
                        self._say(f"  -> {output[:120]}")
                    results.append((call.name, output))
                messages.append(self.llm.tool_results_message(results))
        except LLMError as exc:
            logger.error("LLM failure: %s", exc)
            return f"Sorry, I could not reach the language model: {exc}", trace
        except Exception as exc:                     # last resort: the CLI must never crash
            logger.exception("Unexpected agent failure")
            return f"Sorry, something went wrong: {exc}", trace
        return "I could not finish the answer within the step limit. Try a narrower question.", trace
