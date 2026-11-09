"""Thin wrapper around the Gemini API: function calling, planning, and retries."""
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from config import DEFAULT_MODEL
from tools.registry import ToolSpec

logger = logging.getLogger(__name__)
RETRY_WAITS = (2, 4, 8)                    # seconds; free tier allows about 15 requests a minute
RETRYABLE = {429, 500, 502, 503, 504}
TIMEOUT_MS = 60_000                        # never wait more than a minute for one model call


class LLMError(Exception):
    """The language model could not be reached after retries."""


@dataclass
class FunctionCall:
    name: str
    args: dict


@dataclass
class Reply:
    content: Any                           # the model message, appended to the conversation as is
    text: str = ""
    function_calls: list[FunctionCall] = field(default_factory=list)


class GeminiLLM:
    def __init__(self, system_prompt: str, tool_specs: list[ToolSpec]) -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                                  http_options=types.HttpOptions(timeout=TIMEOUT_MS))
        self.model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        declarations = [
            types.FunctionDeclaration(
                name=spec.name, description=spec.description,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={spec.param: types.Schema(type=types.Type.STRING)},
                    required=[spec.param]))
            for spec in tool_specs]
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0)

    # --- messages -------------------------------------------------------------------------
    def user_message(self, text: str) -> types.Content:
        return types.Content(role="user", parts=[types.Part(text=text)])

    def tool_results_message(self, results: list[tuple[str, str]]) -> types.Content:
        parts = [types.Part.from_function_response(name=name, response={"result": output})
                 for name, output in results]
        return types.Content(role="user", parts=parts)

    # --- calls ----------------------------------------------------------------------------
    def plan(self, plan_prompt: str) -> str:
        """Planning step: one call without tools."""
        logger.info("Gemini plan call (%s)", self.model)
        response = self._retry(lambda: self.client.models.generate_content(
            model=self.model, contents=plan_prompt,
            config=types.GenerateContentConfig(temperature=0)))
        return (response.text or "").strip()

    def generate(self, messages: list) -> Reply:
        logger.info("Gemini call (%s), %d messages", self.model, len(messages))
        response = self._retry(lambda: self.client.models.generate_content(
            model=self.model, contents=messages, config=self.config))
        if not response.candidates or response.candidates[0].content is None:
            return Reply(content=None, text="(The model returned no answer.)")
        content = response.candidates[0].content
        parts = content.parts or []
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        calls = [FunctionCall(c.name, dict(c.args or {})) for c in (response.function_calls or [])]
        return Reply(content=content, text=text.strip(), function_calls=calls)

    def _retry(self, call):
        for wait in (*RETRY_WAITS, None):
            try:
                return call()
            except errors.APIError as exc:
                if exc.code not in RETRYABLE or wait is None:
                    raise LLMError(f"Gemini API error {exc.code}: {exc.message}") from exc
                logger.warning("Gemini error %s, retrying in %ss", exc.code, wait)
            except (ConnectionError, TimeoutError, httpx.TransportError) as exc:
                if wait is None:
                    raise LLMError(f"Cannot reach the Gemini API: {exc}") from exc
                logger.warning("Network error %s, retrying in %ss", exc, wait)
            time.sleep(wait)
