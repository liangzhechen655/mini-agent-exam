from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Message:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }
        if self.name:
            data["name"] = self.name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=data.get("tool_calls", []),
            created_at=data.get("created_at", now_iso()),
        )


@dataclass
class TraceEvent:
    event: str
    detail: dict[str, Any]
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "detail": self.detail,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceEvent":
        return cls(
            event=data["event"],
            detail=data.get("detail", {}),
            created_at=data.get("created_at", now_iso()),
        )


@dataclass
class Session:
    session_id: str
    user_id: str
    window_id: str
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    status: str = "idle"
    state: dict[str, Any] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    updated_at: str = field(default_factory=now_iso)

    def add_message(
        self,
        role: str,
        content: str,
        name: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.messages.append(
            Message(
                role=role,
                content=content,
                name=name,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls or [],
            )
        )
        self.updated_at = now_iso()

    def add_trace(self, event: str, detail: dict[str, Any]) -> None:
        self.trace.append(TraceEvent(event=event, detail=detail))
        self.updated_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "window_id": self.window_id,
            "messages": [message.to_dict() for message in self.messages],
            "summary": self.summary,
            "status": self.status,
            "state": self.state,
            "trace": [event.to_dict() for event in self.trace],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            window_id=data["window_id"],
            messages=[Message.from_dict(item) for item in data.get("messages", [])],
            summary=data.get("summary", ""),
            status=data.get("status", "idle"),
            state=data.get("state", {}),
            trace=[TraceEvent.from_dict(item) for item in data.get("trace", [])],
            updated_at=data.get("updated_at", now_iso()),
        )
