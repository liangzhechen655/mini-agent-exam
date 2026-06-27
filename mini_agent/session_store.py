from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Session


class JsonSessionStore:
    """Persist each session as one JSON file."""

    def __init__(self, root: str | Path = ".agent_data/sessions") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, user_id: str, window_id: str) -> Session:
        session_id = self.session_id(user_id, window_id)
        path = self._path(session_id)
        if not path.exists():
            return Session(session_id=session_id, user_id=user_id, window_id=window_id)
        with path.open("r", encoding="utf-8") as file:
            return Session.from_dict(json.load(file))

    def save(self, session: Session) -> None:
        path = self._path(session.session_id)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(session.to_dict(), file, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def list(self, user_id: str | None = None) -> list[Session]:
        sessions: list[Session] = []
        for path in sorted(self.root.glob("*.json")):
            with path.open("r", encoding="utf-8") as file:
                session = Session.from_dict(json.load(file))
            if user_id is None or session.user_id == user_id:
                sessions.append(session)
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def session_id(self, user_id: str, window_id: str) -> str:
        return f"{self._safe(user_id)}__{self._safe(window_id)}"

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def _safe(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "default"
