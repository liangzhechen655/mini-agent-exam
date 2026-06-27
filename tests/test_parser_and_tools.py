import unittest

from mini_agent.llm import parse_llm_message, parse_llm_output, parse_text_tool_call
from mini_agent.models import Session
from mini_agent.tools import build_default_tools


class ParserAndToolsTest(unittest.TestCase):
    def test_parse_tool_call_from_markdown_json(self) -> None:
        output = """```json
        {"thought":"需要计算","action":"tool","tool_call":{"name":"calculator","arguments":{"expression":"1+2*3"}}}
        ```"""
        parsed = parse_llm_output(output)
        self.assertEqual(parsed.action, "tool")
        self.assertEqual(parsed.tool_name, "calculator")
        self.assertEqual(parsed.arguments, {"expression": "1+2*3"})

    def test_parse_final_answer(self) -> None:
        parsed = parse_llm_output('{"thought":"信息足够","action":"final","final":"答案是 7"}')
        self.assertEqual(parsed.action, "final")
        self.assertEqual(parsed.final, "答案是 7")

    def test_parse_json_with_prefix_and_suffix(self) -> None:
        parsed = parse_llm_output('好的，结果如下：{"thought":"ok","action":"final","final":"完成"}谢谢')
        self.assertEqual(parsed.action, "final")
        self.assertEqual(parsed.final, "完成")

    def test_parse_native_openai_tool_calls(self) -> None:
        parsed = parse_llm_message(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city":"上海","date":"today"}',
                        },
                    }
                ],
            }
        )
        self.assertEqual(parsed.action, "tool")
        self.assertEqual(parsed.tool_name, "weather")
        self.assertEqual(parsed.tool_calls[0]["id"], "call_1")
        self.assertEqual(parsed.arguments, {"city": "上海", "date": "today"})

    def test_parse_text_tool_call_fallback(self) -> None:
        parsed = parse_text_tool_call(
            '我需要查一下。<tool_call>{"name":"search","arguments":{"query":"Agent Runtime"}}</tool_call>'
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.tool_name, "search")
        self.assertEqual(parsed.arguments, {"query": "Agent Runtime"})

    def test_calculator_tool(self) -> None:
        tools = build_default_tools()
        session = Session(session_id="s", user_id="u", window_id="w")
        result = tools.execute("calculator", {"expression": "(12 + 8) / 5"}, session)
        self.assertEqual(result["result"], 4)

    def test_calculator_supports_math_functions(self) -> None:
        tools = build_default_tools()
        session = Session(session_id="s", user_id="u", window_id="w")
        result = tools.execute("calculator", {"expression": "sqrt(16) + sin(pi / 2)"}, session)
        self.assertEqual(result["result"], 5)

    def test_tool_argument_validation_failure(self) -> None:
        tools = build_default_tools()
        session = Session(session_id="s", user_id="u", window_id="w")
        with self.assertRaises(ValueError):
            tools.execute("search", {"query": "Agent", "top_k": "3"}, session)

    def test_weather_search_and_read_docs_tools(self) -> None:
        tools = build_default_tools()
        session = Session(session_id="s", user_id="u", window_id="w")
        weather = tools.execute("weather", {"city": "上海", "date": "today"}, session)
        search = tools.execute("search", {"query": "Agent Runtime", "top_k": 1}, session)
        docs = tools.execute("read_docs", {"query": "runtime"}, session)
        self.assertIn("上海", weather["city"])
        self.assertTrue(search["results"])
        self.assertIn("matches", docs)

    def test_todo_tool_mutates_only_current_session(self) -> None:
        tools = build_default_tools()
        session = Session(session_id="s", user_id="u", window_id="w")
        tools.execute("todo", {"action": "add", "text": "查天气"}, session)
        result = tools.execute("todo", {"action": "list"}, session)
        self.assertEqual(result["todos"][0]["text"], "查天气")

    def test_registry_can_emit_openai_tool_schema(self) -> None:
        tools = build_default_tools()
        openai_tools = tools.openai_tools()
        self.assertEqual(openai_tools[0]["type"], "function")
        self.assertIn("parameters", openai_tools[0]["function"])


if __name__ == "__main__":
    unittest.main()
