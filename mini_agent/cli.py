from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runtime import AgentRuntime
from .session_store import JsonSessionStore


def safe_print(value: object = "") -> None:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _next_window_id(existing: list[str]) -> str:
    index = 1
    while f"window{index}" in existing:
        index += 1
    return f"window{index}"


def _print_help() -> None:
    print(
        """
命令：
  /new              创建并切换到一个新窗口 session
  /switch <window>  切换到指定窗口，例如 /switch window2
  /list             列出当前用户的所有 session
  /history          查看当前 session 的消息历史
  /state            查看当前 session 的结构化状态
  /trace            查看上一轮 trace
  /help             查看帮助
  /exit             退出
""".strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Agent Runtime CLI")
    parser.add_argument("--user", default="userA", help="用户 id")
    parser.add_argument("--window", default="window1", help="窗口/session id")
    parser.add_argument("--data-dir", default=".agent_data/sessions", help="session 存储目录")
    parser.add_argument("--once", help="只发送一条消息后退出")
    args = parser.parse_args()

    runtime = AgentRuntime(store=JsonSessionStore(Path(args.data_dir)))
    user_id = args.user
    window_id = args.window
    if args.once:
        response = runtime.chat(user_id, window_id, args.once)
        safe_print(response.answer)
        return

    safe_print(f"Minimal Agent CLI. user={user_id}, window={window_id}")
    safe_print("输入 /help 查看命令，输入 /exit 退出。")
    last_trace = []
    while True:
        user_input = input(f"{user_id}/{window_id}> ").strip()
        if user_input in {"/exit", "exit", "quit"}:
            break
        if user_input == "/help":
            _print_help()
            continue
        if user_input == "/new":
            existing = [session.window_id for session in runtime.store.list(user_id)]
            window_id = _next_window_id(existing)
            runtime.store.save(runtime.store.get(user_id, window_id))
            safe_print(f"已创建并切换到 {window_id}")
            continue
        if user_input.startswith("/switch"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                safe_print("用法：/switch <window>")
                continue
            window_id = parts[1].strip()
            runtime.store.save(runtime.store.get(user_id, window_id))
            safe_print(f"已切换到 {window_id}")
            continue
        if user_input == "/list":
            sessions = runtime.store.list(user_id)
            if not sessions:
                safe_print("(暂无 session)")
                continue
            for session in sessions:
                marker = "*" if session.window_id == window_id else " "
                turns = sum(1 for message in session.messages if message.role == "user")
                safe_print(f"{marker} {session.window_id} | turns={turns} | messages={len(session.messages)} | updated={session.updated_at}")
            continue
        if user_input == "/history":
            session = runtime.store.get(user_id, window_id)
            if not session.messages:
                safe_print("(暂无历史)")
                continue
            for index, message in enumerate(session.messages):
                name = f":{message.name}" if message.name else ""
                preview = message.content.replace("\n", " ")[:180]
                safe_print(f"[{index}] {message.role}{name}: {preview}")
            continue
        if user_input == "/state":
            session = runtime.store.get(user_id, window_id)
            safe_print(json.dumps(session.state, ensure_ascii=False, indent=2))
            continue
        if user_input == "/trace":
            safe_print(json.dumps(last_trace, ensure_ascii=False, indent=2))
            continue
        if not user_input:
            continue
        response = runtime.chat(user_id, window_id, user_input)
        last_trace = response.trace
        safe_print(response.answer)


if __name__ == "__main__":
    main()
