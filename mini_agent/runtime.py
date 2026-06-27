from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import LLMProvider, OpenAICompatibleLLM, ParsedOutput, parse_llm_message, parse_llm_output
from .models import Message, Session
from .session_store import JsonSessionStore
from .tools import ToolRegistry, build_default_tools


SYSTEM_PROMPT = """你是一个最小可用 Agent Runtime 里的决策模型。
你必须只输出一个 JSON 对象，不要输出 Markdown，不要输出额外解释。

输出格式二选一：
1. 直接回复用户：
{"thought":"为什么可以直接回答，简短写","action":"final","final":"给用户的最终回复"}

2. 调用工具：
{"thought":"为什么需要这个工具，简短写","action":"tool","tool_call":{"name":"工具名","arguments":{}}}

决策原则：
- 需要计算、搜索、查天气、读文档、管理待办时优先调用工具。
- 工具返回后，根据结果继续判断。若信息足够，给 final。
- thought 只用于 trace，请保持简短，不要放隐私或长推理。
- 不要编造工具结果。
"""


@dataclass
class AgentResponse:
    answer: str
    session_id: str
    trace: list[dict[str, Any]] = field(default_factory=list)


class AgentRuntime:
    def __init__(
        self,
        llm: LLMProvider | None = None,
        tools: ToolRegistry | None = None,
        store: JsonSessionStore | None = None,
        max_loop_steps: int = 6,
        max_recent_messages: int = 12,
        compress_threshold: int = 18,
    ) -> None:
        self.llm = llm or OpenAICompatibleLLM()
        self.tools = tools or build_default_tools(docs_dir=Path(__file__).resolve().parents[1] / "docs")
        self.store = store or JsonSessionStore(Path.cwd() / ".agent_data" / "sessions")
        self.max_loop_steps = max_loop_steps
        self.max_recent_messages = max_recent_messages
        self.compress_threshold = compress_threshold

    def chat(self, user_id: str, window_id: str, user_input: str) -> AgentResponse:
        session = self.store.get(user_id=user_id, window_id=window_id)
        if session.status == "busy":
            return AgentResponse(
                answer="当前 session 正在处理中，请稍后再发。后续异步版可以把这条消息放入 session event queue。",
                session_id=session.session_id,
                trace=[],
            )
        session.status = "busy"
        session.add_message("user", user_input)
        turn_trace_start = len(session.trace)

        answer = ""
        try:
            for step in range(1, self.max_loop_steps + 1):
                self._compress_if_needed(session)
                prompt_messages = self._build_prompt_messages(session)
                try:
                    parsed = self._request_llm_decision_with_retry(prompt_messages, session, step)
                except Exception as exc:
                    session.add_trace("llm_parse_error", {"step": step, "error": str(exc)})
                    answer = f"抱歉，我这一步没有得到可解析的模型输出：{exc}"
                    session.add_message("assistant", answer)
                    break

                session.add_trace(
                    "llm_decision",
                    {
                        "step": step,
                        "thought": parsed.thought,
                        "action": parsed.action,
                        "tool": parsed.tool_name,
                        "arguments": parsed.arguments,
                        "tool_calls": parsed.iter_tool_calls(),
                    },
                )

                if parsed.action == "final":
                    answer = parsed.final or ""
                    session.add_message("assistant", answer)
                    break

                if parsed.action == "tool":
                    tool_calls = self._normalise_tool_calls(parsed.iter_tool_calls())
                    session.add_message(
                        "assistant",
                        parsed.thought,
                        tool_calls=[self._to_openai_tool_call(call) for call in tool_calls],
                    )
                    for tool_call in tool_calls:
                        result = self._execute_tool(
                            session,
                            tool_call.get("name", ""),
                            tool_call.get("arguments", {}),
                            step,
                            tool_call_id=tool_call.get("id", ""),
                        )
                        tool_content = json.dumps(result, ensure_ascii=False)
                        session.add_message(
                            "tool",
                            tool_content,
                            name=tool_call.get("name"),
                            tool_call_id=tool_call.get("id", ""),
                        )
                    continue
            else:
                answer = "我已经达到本轮最大工具调用次数，先把目前信息停在这里。你可以继续追问，我会接着处理。"
                session.add_message("assistant", answer)
                session.add_trace("loop_stopped", {"reason": "max_loop_steps", "max_loop_steps": self.max_loop_steps})
        finally:
            session.status = "idle"
            self._compress_if_needed(session)
            self.store.save(session)
        return AgentResponse(
            answer=answer,
            session_id=session.session_id,
            trace=[event.to_dict() for event in session.trace[turn_trace_start:]],
        )

    def _request_llm_decision_with_retry(
        self,
        messages: list[dict[str, Any]],
        session: Session,
        step: int,
        retries: int = 1,
    ) -> ParsedOutput:
        last_error: Exception | None = None
        current_messages = messages
        for attempt in range(retries + 1):
            try:
                return self._request_llm_decision(current_messages)
            except Exception as exc:
                last_error = exc
                session.add_trace(
                    "llm_retry",
                    {"step": step, "attempt": attempt + 1, "error": str(exc)},
                )
                current_messages = messages + [
                    {
                        "role": "system",
                        "content": "上一轮输出无法解析。请严格输出合法 JSON，或使用原生 tool_calls，不要添加 Markdown 解释。",
                    }
                ]
        raise last_error or RuntimeError("LLM decision failed")

    def _normalise_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalised = []
        for call in tool_calls:
            normalised.append(
                {
                    "id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "name": call.get("name", ""),
                    "arguments": call.get("arguments", {}),
                }
            )
        return normalised

    def _to_openai_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": call["id"],
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
            },
        }

    def _request_llm_decision(self, messages: list[dict[str, Any]]) -> ParsedOutput:
        complete_message = getattr(self.llm, "complete_message", None)
        if callable(complete_message):
            message = complete_message(messages, tools=self.tools.openai_tools())
            return parse_llm_message(message)
        raw_output = self.llm.complete(messages)
        return parse_llm_output(raw_output)

    def _execute_tool(
        self,
        session: Session,
        name: str,
        arguments: dict[str, Any],
        step: int,
        tool_call_id: str = "",
    ) -> dict[str, Any]:
        try:
            result = self.tools.execute(name, arguments, session)
            session.add_trace(
                "tool_result",
                {
                    "step": step,
                    "tool_call_id": tool_call_id,
                    "tool": name,
                    "arguments": arguments,
                    "result": result,
                },
            )
            return {"ok": True, "tool": name, "result": result}
        except Exception as exc:
            error = {"ok": False, "tool": name, "tool_call_id": tool_call_id, "error": str(exc), "arguments": arguments}
            session.add_trace("tool_error", {"step": step, **error})
            return error

    def _build_prompt_messages(self, session: Session) -> list[dict[str, Any]]:
        system_parts = [
            SYSTEM_PROMPT,
            "可用工具 schema：",
            self.tools.specs_json(),
        ]
        if session.summary:
            system_parts.extend(["历史会话摘要：", session.summary])
        if session.state:
            system_parts.extend(["当前 session state：", json.dumps(session.state, ensure_ascii=False)])

        messages = [{"role": "system", "content": "\n".join(system_parts)}]
        for message in self._recent_messages(session):
            if message.role == "tool":
                if not message.tool_call_id:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[历史工具结果 {message.name}] {message.content}",
                        }
                    )
                    continue
                data = {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id,
                }
                messages.append(data)
            elif message.role in {"user", "assistant"}:
                data = {"role": message.role, "content": message.content}
                if message.tool_calls:
                    data["tool_calls"] = message.tool_calls
                messages.append(data)
        return messages

    def _recent_messages(self, session: Session) -> list[Message]:
        messages = session.messages
        if len(messages) <= self.max_recent_messages:
            return messages
        start = len(messages) - self.max_recent_messages
        if messages[start].role == "tool" and start > 0 and messages[start - 1].role == "assistant":
            start -= 1
        return messages[start:]

    def _compress_if_needed(self, session: Session) -> None:
        if len(session.messages) <= self.compress_threshold:
            return
        keep = session.messages[-self.max_recent_messages :]
        old = session.messages[: -self.max_recent_messages]
        summary = self._generate_summary(old, session.summary)
        merged = []
        if session.summary:
            merged.append(session.summary.strip())
        if summary:
            merged.append("本次压缩新增摘要：\n" + summary)
        session.summary = "\n".join(part for part in merged if part).strip()[-4000:]
        session.messages = keep
        session.add_trace(
            "context_compressed",
            {
                "old_message_count": len(old),
                "kept_message_count": len(keep),
                "summary_chars": len(session.summary),
            },
        )

    def _generate_summary(self, messages: list[Message], previous_summary: str = "") -> str:
        formatted = self._format_messages_for_summary(messages)
        if not formatted:
            return ""
        complete_message = getattr(self.llm, "complete_message", None)
        if callable(complete_message):
            prompt = (
                "请把下面的 Agent 会话历史压缩成一段可继续对话的中文工作摘要。\n"
                "必须保留：用户目标、关键事实、工具调用结果、待办状态、未完成事项、重要指代锚点。\n"
                "不要写无关寒暄，不要泄露隐藏推理。控制在 300 字以内。\n\n"
            )
            if previous_summary:
                prompt += f"已有摘要：\n{previous_summary}\n\n"
            prompt += f"待压缩历史：\n{formatted}"
            try:
                message = complete_message(
                    [{"role": "user", "content": prompt}],
                    tools=None,
                )
                content = (message.get("content") or "").strip()
                if content:
                    return content[:1200]
            except Exception:
                pass
        return "\n".join(self._extract_summary_bullets(messages))

    def _format_messages_for_summary(self, messages: list[Message]) -> str:
        lines: list[str] = []
        for message in messages:
            content = message.content.replace("\n", " ").strip()
            if not content:
                continue
            if message.role == "tool":
                lines.append(f"[工具 {message.name} 返回] {content[:500]}")
            elif message.role == "assistant" and message.tool_calls:
                names = [call.get("function", {}).get("name", "?") for call in message.tool_calls]
                lines.append(f"[Agent 请求工具] {', '.join(names)}")
                if content:
                    lines.append(f"[Agent thought] {content[:200]}")
            elif message.role == "assistant":
                lines.append(f"[Agent] {content[:300]}")
            elif message.role == "user":
                lines.append(f"[用户] {content[:300]}")
        return "\n".join(lines)

    def _extract_summary_bullets(self, messages: list[Message]) -> list[str]:
        bullets: list[str] = []
        for message in messages:
            content = message.content.replace("\n", " ").strip()
            if not content:
                continue
            if message.role == "user":
                bullets.append(f"- 用户说：{content[:180]}")
            elif message.role == "assistant":
                bullets.append(f"- Agent 回复：{content[:180]}")
            elif message.role == "tool":
                bullets.append(f"- 工具 {message.name} 结果：{content[:220]}")
        return bullets[-20:]
