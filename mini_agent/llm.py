from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs without adding a third-party dependency."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class LLMProvider(Protocol):
    def complete(self, messages: list[dict[str, Any]]) -> str:
        ...


class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible chat/completions client using only stdlib."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        timeout: int = 60,
    ) -> None:
        load_dotenv()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("missing OPENAI_API_KEY or LLM_API_KEY")

    def complete(self, messages: list[dict[str, Any]]) -> str:
        message = self.complete_message(messages)
        return message.get("content") or ""

    def complete_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        return payload["choices"][0]["message"]


@dataclass
class ParsedOutput:
    thought: str
    action: str
    final: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def iter_tool_calls(self) -> list[dict[str, Any]]:
        if self.tool_calls:
            return self.tool_calls
        if self.tool_name:
            return [{"id": "", "name": self.tool_name, "arguments": self.arguments or {}}]
        return []


def parse_llm_output(text: str) -> ParsedOutput:
    data = _load_json_object(text)
    thought = str(data.get("thought") or data.get("thinking") or "")
    action = str(data.get("action") or "").lower()
    if action == "final":
        return ParsedOutput(thought=thought, action="final", final=str(data.get("final") or data.get("answer") or ""))
    if action == "tool":
        calls = data.get("tool_calls")
        if isinstance(calls, list):
            parsed_calls = [_parse_json_tool_call(call) for call in calls]
            first = parsed_calls[0] if parsed_calls else {}
            return ParsedOutput(
                thought=thought,
                action="tool",
                tool_name=first.get("name"),
                arguments=first.get("arguments"),
                tool_calls=parsed_calls,
            )
        call = data.get("tool_call") or data.get("tool") or {}
        parsed_call = _parse_json_tool_call(call)
        return ParsedOutput(
            thought=thought,
            action="tool",
            tool_name=parsed_call["name"],
            arguments=parsed_call["arguments"],
            tool_calls=[parsed_call],
        )
    raise ValueError(f"unsupported LLM action: {action}")


def parse_llm_message(message: dict[str, Any]) -> ParsedOutput:
    """Parse either native OpenAI-compatible tool_calls or text JSON fallback."""

    native_tool_calls = message.get("tool_calls") or []
    thought = str(message.get("reasoning_content") or message.get("thinking") or message.get("content") or "")
    if native_tool_calls:
        parsed_calls = []
        for call in native_tool_calls:
            function = call.get("function", {})
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            parsed_calls.append(
                {
                    "id": call.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": arguments,
                }
            )
        first = parsed_calls[0]
        return ParsedOutput(
            thought=thought.strip(),
            action="tool",
            tool_name=first["name"],
            arguments=first["arguments"],
            tool_calls=parsed_calls,
        )

    content = message.get("content") or ""
    try:
        return parse_llm_output(content)
    except Exception:
        fallback = parse_text_tool_call(content)
        if fallback:
            return fallback
        return ParsedOutput(thought="", action="final", final=content.strip())


def parse_text_tool_call(content: str) -> ParsedOutput | None:
    """Fallback parser for models that emit <tool_call>{...}</tool_call> text."""

    matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, flags=re.DOTALL)
    if not matches:
        return None
    calls = []
    for match in matches:
        try:
            calls.append(_parse_json_tool_call(json.loads(match.strip())))
        except Exception:
            continue
    if not calls:
        return None
    thought = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
    first = calls[0]
    return ParsedOutput(
        thought=thought,
        action="tool",
        tool_name=first["name"],
        arguments=first["arguments"],
        tool_calls=calls,
    )


def _parse_json_tool_call(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise ValueError("tool_call should be an object")
    name = call.get("name")
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        arguments = _load_json_object(arguments)
    if not name:
        raise ValueError("tool action missing tool_call.name")
    if not isinstance(arguments, dict):
        raise ValueError("tool_call.arguments should be an object")
    return {"id": call.get("id", ""), "name": name, "arguments": arguments}


def _load_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    data = _decode_first_json_object(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM output JSON should be an object")
    return data


def _decode_first_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
            return data
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM output does not contain a JSON object")
