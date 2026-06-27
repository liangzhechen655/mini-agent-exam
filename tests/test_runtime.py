import tempfile
import unittest
from pathlib import Path

from mini_agent.runtime import AgentRuntime
from mini_agent.session_store import JsonSessionStore
from mini_agent.tools import build_default_tools


class FakeLLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.index = 0
        self.seen_messages = []

    def complete(self, messages: list[dict]) -> str:
        self.seen_messages.append(messages)
        if self.index >= len(self.outputs):
            raise AssertionError("FakeLLM has no more outputs")
        output = self.outputs[self.index]
        self.index += 1
        return output


class FakeNativeLLM:
    def __init__(self, outputs: list[dict]) -> None:
        self.outputs = outputs
        self.index = 0
        self.seen_tools = []
        self.seen_messages = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        raise AssertionError("native fake should use complete_message")

    def complete_message(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.seen_tools.append(tools or [])
        self.seen_messages.append(messages)
        if self.index >= len(self.outputs):
            raise AssertionError("FakeNativeLLM has no more outputs")
        output = self.outputs[self.index]
        self.index += 1
        return output


class RuntimeTest(unittest.TestCase):
    def make_runtime(self, tmpdir: str, outputs: list[str], **kwargs) -> AgentRuntime:
        return AgentRuntime(
            llm=FakeLLM(outputs),
            tools=build_default_tools(docs_dir=Path(tmpdir)),
            store=JsonSessionStore(Path(tmpdir) / "sessions"),
            **kwargs,
        )

    def test_tool_loop_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self.make_runtime(
                tmpdir,
                [
                    '{"thought":"需要计算","action":"tool","tool_call":{"name":"calculator","arguments":{"expression":"2+5"}}}',
                    '{"thought":"工具结果足够","action":"final","final":"2+5=7"}',
                ],
            )
            response = runtime.chat("A", "window1", "帮我算 2+5")
            self.assertEqual(response.answer, "2+5=7")
            self.assertEqual(response.trace[0]["event"], "llm_decision")
            self.assertEqual(response.trace[1]["event"], "tool_result")

    def test_native_tool_calling_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = FakeNativeLLM(
                [
                    {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expression":"10/2"}',
                                },
                            }
                        ],
                    },
                    {"content": "10/2=5"},
                ]
            )
            runtime = AgentRuntime(
                llm=fake,
                tools=build_default_tools(docs_dir=Path(tmpdir)),
                store=JsonSessionStore(Path(tmpdir) / "sessions"),
            )
            response = runtime.chat("A", "window1", "帮我算 10/2")
            self.assertEqual(response.answer, "10/2=5")
            self.assertTrue(fake.seen_tools[0])
            self.assertEqual(response.trace[1]["detail"]["tool_call_id"], "call_1")
            second_prompt = fake.seen_messages[1]
            self.assertTrue(any(message["role"] == "tool" and message["tool_call_id"] == "call_1" for message in second_prompt))
            self.assertTrue(any(message["role"] == "assistant" and message.get("tool_calls") for message in second_prompt))

    def test_invalid_json_retries_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self.make_runtime(
                tmpdir,
                [
                    "这不是 JSON",
                    '{"thought":"重试后格式正确","action":"final","final":"已恢复"}',
                ],
            )
            response = runtime.chat("A", "window1", "测试坏输出")
            self.assertEqual(response.answer, "已恢复")
            self.assertTrue(any(event["event"] == "llm_retry" for event in response.trace))

    def test_multi_step_tool_chain_weather_then_todo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self.make_runtime(
                tmpdir,
                [
                    '{"thought":"先查天气","action":"tool","tool_call":{"name":"weather","arguments":{"city":"上海","date":"today"}}}',
                    '{"thought":"下雨，需要记待办","action":"tool","tool_call":{"name":"todo","arguments":{"action":"add","text":"出门带伞"}}}',
                    '{"thought":"两个工具都完成","action":"final","final":"上海今天小雨，已记录待办：出门带伞。"}',
                ],
            )
            response = runtime.chat("A", "window1", "查上海天气，如果下雨就记待办")
            self.assertIn("出门带伞", response.answer)
            session = runtime.store.get("A", "window1")
            self.assertEqual(session.state["todos"][0]["text"], "出门带伞")

    def test_follow_up_with_tool_uses_previous_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = FakeLLM(
                [
                    '{"thought":"查今天上海天气","action":"tool","tool_call":{"name":"weather","arguments":{"city":"上海","date":"today"}}}',
                    '{"thought":"回答今天","action":"final","final":"上海今天小雨。"}',
                    '{"thought":"追问明天，沿用上海","action":"tool","tool_call":{"name":"weather","arguments":{"city":"上海","date":"tomorrow"}}}',
                    '{"thought":"回答明天","action":"final","final":"上海明天阴。"}',
                ],
            )
            runtime = AgentRuntime(
                llm=fake,
                tools=build_default_tools(docs_dir=Path(tmpdir)),
                store=JsonSessionStore(Path(tmpdir) / "sessions"),
            )
            runtime.chat("A", "window1", "上海今天天气？")
            response = runtime.chat("A", "window1", "那明天呢？")
            self.assertEqual(response.answer, "上海明天阴。")
            second_turn_prompt = fake.seen_messages[2]
            self.assertTrue(any("last_weather_city" in message["content"] for message in second_turn_prompt if message["role"] == "system"))

    def test_sessions_are_isolated_by_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self.make_runtime(
                tmpdir,
                [
                    '{"thought":"记录窗口1待办","action":"tool","tool_call":{"name":"todo","arguments":{"action":"add","text":"查天气"}}}',
                    '{"thought":"已记录","action":"final","final":"窗口1已记录查天气"}',
                    '{"thought":"记录窗口2待办","action":"tool","tool_call":{"name":"todo","arguments":{"action":"add","text":"写周报"}}}',
                    '{"thought":"已记录","action":"final","final":"窗口2已记录写周报"}',
                    '{"thought":"查询窗口1待办","action":"tool","tool_call":{"name":"todo","arguments":{"action":"list"}}}',
                    '{"thought":"工具结果足够","action":"final","final":"窗口1只有：查天气"}',
                ],
            )
            runtime.chat("A", "window1", "记一个待办：查天气")
            runtime.chat("A", "window2", "记一个待办：写周报")
            response = runtime.chat("A", "window1", "我这个窗口有哪些待办？")

            self.assertEqual(response.answer, "窗口1只有：查天气")
            session1 = runtime.store.get("A", "window1")
            session2 = runtime.store.get("A", "window2")
            self.assertEqual(session1.state["todos"][0]["text"], "查天气")
            self.assertEqual(session2.state["todos"][0]["text"], "写周报")
            listed = runtime.store.list("A")
            self.assertEqual({session.window_id for session in listed}, {"window1", "window2"})

    def test_context_compression_keeps_recent_messages_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = ['{"thought":"直接回答","action":"final","final":"ok"}'] * 8
            runtime = self.make_runtime(
                tmpdir,
                outputs,
                max_recent_messages=4,
                compress_threshold=6,
            )
            for index in range(8):
                runtime.chat("A", "window1", f"第 {index} 轮")

            session = runtime.store.get("A", "window1")
            self.assertLessEqual(len(session.messages), 4)
            self.assertIn("本次压缩新增摘要", session.summary)
            self.assertTrue(any(event.event == "context_compressed" for event in session.trace))


if __name__ == "__main__":
    unittest.main()
