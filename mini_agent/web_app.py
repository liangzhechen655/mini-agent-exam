from __future__ import annotations

import argparse
import json
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .runtime import AgentRuntime
from .session_store import JsonSessionStore


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mini Agent Runtime</title>
  <style>
    :root {
      --bg: #f7f8fb;
      --panel: #ffffff;
      --line: #dfe3ea;
      --text: #172033;
      --muted: #667085;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --danger: #b42318;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
      letter-spacing: 0;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 260px minmax(420px, 1fr) 360px;
    }
    aside, main, section {
      min-width: 0;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }
    aside {
      display: flex;
      flex-direction: column;
    }
    .brand {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
    }
    .user-row {
      display: flex;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    input, textarea, button {
      font: inherit;
    }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      color: var(--text);
      background: #fff;
      outline: none;
    }
    textarea {
      min-height: 46px;
      max-height: 120px;
      resize: vertical;
    }
    input:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 9px 12px;
      cursor: pointer;
      white-space: nowrap;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    button:hover { filter: brightness(0.98); }
    .sessions {
      overflow: auto;
      padding: 8px;
    }
    .session {
      width: 100%;
      text-align: left;
      margin: 4px 0;
      display: block;
      border-radius: 6px;
      box-shadow: none;
    }
    .session.active {
      border-color: rgba(15, 118, 110, 0.35);
      background: #ecfdf5;
    }
    .session strong {
      display: block;
      font-size: 14px;
      line-height: 20px;
    }
    .session span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 18px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    main {
      display: flex;
      flex-direction: column;
      background: #fbfcfe;
    }
    .topbar {
      height: 64px;
      padding: 0 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .title {
      min-width: 0;
    }
    .title h1 {
      margin: 0;
      font-size: 18px;
      line-height: 24px;
    }
    .title p {
      margin: 2px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 18px;
    }
    .messages {
      flex: 1;
      overflow: auto;
      padding: 18px;
    }
    .msg {
      max-width: 820px;
      margin: 0 0 12px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: var(--shadow);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.55;
    }
    .msg.user {
      margin-left: auto;
      border-color: rgba(37, 99, 235, 0.28);
      background: #eff6ff;
    }
    .msg.assistant {
      margin-right: auto;
    }
    .msg.tool {
      font-size: 12px;
      color: #475467;
      background: #f8fafc;
      box-shadow: none;
    }
    .composer {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 14px 18px 18px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }
    section {
      border-right: 0;
      display: flex;
      flex-direction: column;
    }
    .tabs {
      height: 64px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 0 12px;
      border-bottom: 1px solid var(--line);
    }
    .tabs button.active {
      border-color: var(--accent-2);
      color: var(--accent-2);
      background: #eff6ff;
    }
    pre {
      margin: 0;
      padding: 14px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 12px;
      line-height: 1.5;
    }
    .status {
      color: var(--muted);
      font-size: 12px;
    }
    .error { color: var(--danger); }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      aside, section { min-height: 220px; border-right: 0; border-bottom: 1px solid var(--line); }
      main { min-height: 560px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <span>Mini Agent</span>
        <button id="newSession">新窗口</button>
      </div>
      <div class="user-row">
        <input id="userId" value="webdemo" aria-label="user id" />
      </div>
      <div id="sessions" class="sessions"></div>
    </aside>
    <main>
      <div class="topbar">
        <div class="title">
          <h1 id="windowTitle">window1</h1>
          <p id="sessionMeta">ready</p>
        </div>
        <span id="status" class="status">idle</span>
      </div>
      <div id="messages" class="messages"></div>
      <form id="composer" class="composer">
        <textarea id="messageInput" placeholder="输入消息，例如：查上海今天真实天气，如果下雨就记待办"></textarea>
        <button class="primary" type="submit">发送</button>
      </form>
    </main>
    <section>
      <div class="tabs">
        <button id="stateTab" class="active" type="button">State</button>
        <button id="traceTab" type="button">Trace</button>
      </div>
      <pre id="sidePanel">{}</pre>
    </section>
  </div>
  <script>
    const qs = (sel) => document.querySelector(sel);
    let currentWindow = "window1";
    let currentTab = "state";
    let lastTrace = [];

    function userId() {
      return qs("#userId").value.trim() || "webdemo";
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: {"Content-Type": "application/json"},
        ...options
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    function renderMessages(messages) {
      const box = qs("#messages");
      box.innerHTML = "";
      for (const message of messages) {
        appendMessage(message, false);
      }
      box.scrollTop = box.scrollHeight;
    }

    function appendMessage(message, scroll = true) {
      const box = qs("#messages");
      const div = document.createElement("div");
      div.className = `msg ${message.role}`;
      if (message.pending) div.style.opacity = "0.72";
      const label = message.name ? `${message.role}:${message.name}` : message.role;
      div.textContent = `${label}\n${message.content}`;
      box.appendChild(div);
      if (scroll) box.scrollTop = box.scrollHeight;
      return div;
    }

    function renderSide(session) {
      if (currentTab === "trace") {
        qs("#sidePanel").textContent = JSON.stringify(lastTrace.length ? lastTrace : session.trace.slice(-8), null, 2);
      } else {
        qs("#sidePanel").textContent = JSON.stringify(session.state || {}, null, 2);
      }
    }

    async function loadSessions() {
      const data = await api(`/api/sessions?user=${encodeURIComponent(userId())}`);
      const box = qs("#sessions");
      box.innerHTML = "";
      const windows = data.sessions.map((item) => item.window_id);
      if (!windows.includes(currentWindow)) {
        currentWindow = windows[0] || "window1";
      }
      for (const session of data.sessions) {
        const button = document.createElement("button");
        button.className = `session ${session.window_id === currentWindow ? "active" : ""}`;
        button.type = "button";
        button.innerHTML = `<strong>${session.window_id}</strong><span>${session.turns} turns · ${session.status}</span>`;
        button.onclick = () => {
          currentWindow = session.window_id;
          loadAll();
        };
        box.appendChild(button);
      }
      if (!data.sessions.length) {
        await api("/api/session", {
          method: "POST",
          body: JSON.stringify({user_id: userId(), window_id: currentWindow})
        });
        return loadSessions();
      }
    }

    async function loadSession() {
      const session = await api(`/api/session?user=${encodeURIComponent(userId())}&window=${encodeURIComponent(currentWindow)}`);
      qs("#windowTitle").textContent = currentWindow;
      qs("#sessionMeta").textContent = `${session.session_id} · ${session.messages.length} messages`;
      qs("#status").textContent = session.status;
      renderMessages(session.messages);
      renderSide(session);
    }

    async function loadAll() {
      try {
        await loadSessions();
        await loadSession();
        qs("#status").className = "status";
      } catch (error) {
        qs("#status").textContent = error.message;
        qs("#status").className = "status error";
      }
    }

    qs("#composer").addEventListener("submit", async (event) => {
      event.preventDefault();
      const input = qs("#messageInput");
      const message = input.value.trim();
      if (!message) return;
      input.value = "";
      const sendButton = qs("#composer button[type='submit']");
      sendButton.disabled = true;
      appendMessage({role: "user", content: message});
      const pending = appendMessage({role: "assistant", content: "Agent 正在思考…", pending: true});
      qs("#status").textContent = "running";
      try {
        const data = await api("/api/chat", {
          method: "POST",
          body: JSON.stringify({user_id: userId(), window_id: currentWindow, message})
        });
        lastTrace = data.trace || [];
        pending.remove();
        await loadAll();
        currentTab = "trace";
        qs("#traceTab").classList.add("active");
        qs("#stateTab").classList.remove("active");
        await loadSession();
      } catch (error) {
        pending.textContent = `assistant\n请求失败：${error.message}`;
        pending.style.opacity = "1";
        pending.classList.add("error");
        qs("#status").textContent = error.message;
        qs("#status").className = "status error";
      } finally {
        sendButton.disabled = false;
        input.focus();
      }
    });

    qs("#newSession").onclick = async () => {
      const data = await api("/api/session/new", {
        method: "POST",
        body: JSON.stringify({user_id: userId()})
      });
      currentWindow = data.window_id;
      lastTrace = [];
      await loadAll();
    };

    qs("#userId").addEventListener("change", () => {
      currentWindow = "window1";
      lastTrace = [];
      loadAll();
    });
    qs("#stateTab").onclick = async () => {
      currentTab = "state";
      qs("#stateTab").classList.add("active");
      qs("#traceTab").classList.remove("active");
      await loadSession();
    };
    qs("#traceTab").onclick = async () => {
      currentTab = "trace";
      qs("#traceTab").classList.add("active");
      qs("#stateTab").classList.remove("active");
      await loadSession();
    };
    loadAll();
  </script>
</body>
</html>"""


class WebApp:
    def __init__(self, data_dir: Path) -> None:
        self.store = JsonSessionStore(data_dir)
        self.runtime = AgentRuntime(store=self.store)


class Handler(BaseHTTPRequestHandler):
    app: WebApp

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send_text(HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/sessions":
            query = urllib.parse.parse_qs(parsed.query)
            user = query.get("user", ["webdemo"])[0]
            sessions = []
            for session in self.app.store.list(user):
                sessions.append(
                    {
                        "session_id": session.session_id,
                        "user_id": session.user_id,
                        "window_id": session.window_id,
                        "status": session.status,
                        "turns": sum(1 for message in session.messages if message.role == "user"),
                        "updated_at": session.updated_at,
                    }
                )
            self._send_json({"sessions": sessions})
            return
        if parsed.path == "/api/session":
            query = urllib.parse.parse_qs(parsed.query)
            user = query.get("user", ["webdemo"])[0]
            window = query.get("window", ["window1"])[0]
            self._send_json(self._session_payload(user, window))
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        data = self._read_json()
        if parsed.path == "/api/session":
            user = str(data.get("user_id") or "webdemo")
            window = str(data.get("window_id") or "window1")
            session = self.app.store.get(user, window)
            self.app.store.save(session)
            self._send_json(self._session_payload(user, window))
            return
        if parsed.path == "/api/session/new":
            user = str(data.get("user_id") or "webdemo")
            existing = {session.window_id for session in self.app.store.list(user)}
            index = 1
            while f"window{index}" in existing:
                index += 1
            window = f"window{index}"
            session = self.app.store.get(user, window)
            self.app.store.save(session)
            self._send_json({"window_id": window, "session_id": session.session_id})
            return
        if parsed.path == "/api/chat":
            user = str(data.get("user_id") or "webdemo")
            window = str(data.get("window_id") or "window1")
            message = str(data.get("message") or "").strip()
            if not message:
                self._send_json({"error": "message is required"}, HTTPStatus.BAD_REQUEST)
                return
            response = self.app.runtime.chat(user, window, message)
            self._send_json({"answer": response.answer, "session_id": response.session_id, "trace": response.trace})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _session_payload(self, user: str, window: str) -> dict[str, Any]:
        session = self.app.store.get(user, window)
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "window_id": session.window_id,
            "status": session.status,
            "messages": [message.to_dict() for message in session.messages],
            "state": session.state,
            "summary": session.summary,
            "trace": [event.to_dict() for event in session.trace],
        }

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _send_text(self, text: str, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini Agent Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default=".agent_data/sessions")
    args = parser.parse_args()

    Handler.app = WebApp(Path(args.data_dir))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Mini Agent Web UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
