from __future__ import annotations

import ast
import json
import math
import operator
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date as date_cls, timedelta
from pathlib import Path
from typing import Any, Callable

from .models import Session


ToolFunc = Callable[[dict[str, Any], Session], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._funcs: dict[str, ToolFunc] = {}

    def register(self, spec: ToolSpec, func: ToolFunc) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        self._funcs[spec.name] = func

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def specs_json(self) -> str:
        return json.dumps([spec.to_prompt_dict() for spec in self.specs()], ensure_ascii=False, indent=2)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [spec.to_openai_tool() for spec in self.specs()]

    def execute(self, name: str, arguments: dict[str, Any], session: Session) -> dict[str, Any]:
        if name not in self._specs:
            raise ValueError(f"unknown tool: {name}")
        self._validate(name, arguments)
        return self._funcs[name](arguments, session)

    def _validate(self, name: str, arguments: dict[str, Any]) -> None:
        spec = self._specs[name]
        schema = spec.parameters
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for key in required:
            if key not in arguments:
                raise ValueError(f"tool {name} missing required argument: {key}")
        for key, value in arguments.items():
            expected = properties.get(key, {}).get("type")
            if expected and not _type_matches(value, expected):
                raise ValueError(f"tool {name} argument {key} should be {expected}")


def _type_matches(value: Any, expected: str | list[str]) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return any(isinstance(value, mapping[item]) for item in types if item in mapping)


def build_default_tools(docs_dir: str | Path = "docs") -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="计算数学表达式。支持四则运算、括号、幂运算，以及 sqrt/sin/cos/log 等 math 白名单函数。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "例如: sqrt(16) + sin(pi / 2)"}
                },
                "required": ["expression"],
            },
        ),
        calculator_tool,
    )
    registry.register(
        ToolSpec(
            name="search",
            description="搜索 mock 知识库，适合查 Agent、天气、Python、笔试交付等信息。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 3"},
                },
                "required": ["query"],
            },
        ),
        search_tool,
    )
    registry.register(
        ToolSpec(
            name="todo",
            description="管理当前 session 内的待办。支持 add/list/done/remove。",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "add, list, done, remove"},
                    "text": {"type": "string", "description": "新增待办内容"},
                    "id": {"type": "integer", "description": "待办 id"},
                },
                "required": ["action"],
            },
        ),
        todo_tool,
    )
    registry.register(
        ToolSpec(
            name="weather",
            description="查询真实天气。优先调用 Open-Meteo 实时接口，网络失败时才使用 mock fallback。",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "date": {"type": "string", "description": "today/tomorrow 或具体日期"},
                },
                "required": ["city"],
            },
        ),
        weather_tool,
    )
    registry.register(
        ToolSpec(
            name="read_docs",
            description="读取本项目 docs 目录中的说明文档片段。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "想查的关键词"}
                },
                "required": ["query"],
            },
        ),
        make_read_docs_tool(docs_dir),
    )
    return registry


def calculator_tool(arguments: dict[str, Any], session: Session) -> dict[str, Any]:
    expression = arguments["expression"]
    result = _safe_eval(expression)
    return {"expression": expression, "result": result}


def _safe_eval(expression: str) -> float:
    if len(expression) > 500:
        raise ValueError("expression is too long")
    node = ast.parse(expression, mode="eval")
    return _eval_node(node.body)


MATH_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
}


def _eval_node(node: ast.AST) -> float:
    binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
        return binary_ops[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
        return unary_ops[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in MATH_NAMES and isinstance(MATH_NAMES[node.id], (int, float)):
        return MATH_NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in MATH_NAMES:
        func = MATH_NAMES[node.func.id]
        if not callable(func):
            raise ValueError(f"{node.func.id} is not callable")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        return func(*[_eval_node(arg) for arg in node.args])
    raise ValueError("unsupported expression")


MOCK_SEARCH_INDEX = [
    {
        "title": "Agent Runtime 最小闭环",
        "content": "Agent Runtime 通常包含 LLM 决策、工具注册、工具执行、上下文管理、trace 和 session 管理。",
    },
    {
        "title": "笔试交付建议",
        "content": "提交物建议包含代码仓库、README、运行截图或录屏、测试结果、Prompt 与问题解决记录。",
    },
    {
        "title": "天气查询",
        "content": "天气工具通常适合做 mock，重点是验证参数 schema、工具调用和多轮追问。",
    },
    {
        "title": "Context 压缩",
        "content": "长对话可以保留最近窗口、会话摘要、关键事实和工具结果，避免把完整思考过程塞回上下文。",
    },
]


def search_tool(arguments: dict[str, Any], session: Session) -> dict[str, Any]:
    query = arguments["query"].lower()
    top_k = int(arguments.get("top_k", 3))
    scored = []
    for item in MOCK_SEARCH_INDEX:
        haystack = f"{item['title']} {item['content']}".lower()
        score = sum(1 for token in query.split() if token in haystack)
        if query in haystack:
            score += 3
        scored.append((score, item))
    results = [item for score, item in sorted(scored, key=lambda pair: pair[0], reverse=True) if score > 0]
    if not results:
        results = MOCK_SEARCH_INDEX[:top_k]
    return {"query": arguments["query"], "results": results[:top_k]}


def todo_tool(arguments: dict[str, Any], session: Session) -> dict[str, Any]:
    todos = session.state.setdefault("todos", [])
    action = arguments["action"].lower()
    if action == "add":
        text = arguments.get("text", "").strip()
        if not text:
            raise ValueError("todo.add requires text")
        next_id = max([item["id"] for item in todos], default=0) + 1
        item = {"id": next_id, "text": text, "done": False}
        todos.append(item)
        return {"status": "added", "todo": item, "todos": todos}
    if action == "list":
        return {"status": "ok", "todos": todos}
    if action in {"done", "remove"}:
        todo_id = int(arguments.get("id", 0))
        for item in todos:
            if item["id"] == todo_id:
                if action == "done":
                    item["done"] = True
                    return {"status": "done", "todo": item, "todos": todos}
                todos.remove(item)
                return {"status": "removed", "todo": item, "todos": todos}
        raise ValueError(f"todo id not found: {todo_id}")
    raise ValueError(f"unsupported todo action: {action}")


MOCK_WEATHER = {
    "北京": {"today": "晴，-4 到 3 度，空气干冷。", "tomorrow": "多云，-3 到 4 度。"},
    "上海": {"today": "小雨，6 到 10 度，建议带伞。", "tomorrow": "阴，7 到 11 度。"},
    "广州": {"today": "多云，14 到 21 度。", "tomorrow": "晴，15 到 23 度。"},
}

CITY_COORDS = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "杭州": (30.2741, 120.1551),
    "南京": (32.0603, 118.7969),
    "成都": (30.5728, 104.0668),
    "武汉": (30.5928, 114.3055),
    "西安": (34.3416, 108.9398),
    "shanghai": (31.2304, 121.4737),
    "beijing": (39.9042, 116.4074),
    "guangzhou": (23.1291, 113.2644),
}

WEATHER_CODES = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "霜雾",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "较强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    81: "较强阵雨",
    82: "强阵雨",
    95: "雷雨",
}


def weather_tool(arguments: dict[str, Any], session: Session) -> dict[str, Any]:
    city = arguments["city"]
    date = arguments.get("date", "today")
    session.state["last_weather_city"] = city
    try:
        return _live_weather(city, date)
    except Exception as exc:
        city_weather = MOCK_WEATHER.get(city, {})
        forecast = city_weather.get(date, "暂无 mock 数据，默认天气：多云，体感舒适。")
        return {
            "city": city,
            "date": date,
            "forecast": forecast,
            "source": "mock_fallback",
            "fallback_reason": str(exc),
        }


def _live_weather(city: str, date_value: str) -> dict[str, Any]:
    latitude, longitude = _resolve_city(city)
    target_date = _resolve_date(date_value)
    query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "Asia/Shanghai",
            "forecast_days": 3,
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{query}"
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    if target_date in dates:
        index = dates.index(target_date)
        code = int(daily.get("weather_code", [0])[index])
        min_temp = daily.get("temperature_2m_min", [None])[index]
        max_temp = daily.get("temperature_2m_max", [None])[index]
        precipitation = daily.get("precipitation_sum", [None])[index]
        condition = WEATHER_CODES.get(code, f"天气代码 {code}")
        forecast = f"{condition}，{min_temp} 到 {max_temp} 度，降水量 {precipitation} mm。"
        return {
            "city": city,
            "date": date_value,
            "resolved_date": target_date,
            "forecast": forecast,
            "condition": condition,
            "temperature_min_c": min_temp,
            "temperature_max_c": max_temp,
            "precipitation_mm": precipitation,
            "source": "live_open_meteo",
        }

    current = payload.get("current", {})
    code = int(current.get("weather_code", 0))
    condition = WEATHER_CODES.get(code, f"天气代码 {code}")
    temperature = current.get("temperature_2m")
    precipitation = current.get("precipitation")
    forecast = f"{condition}，当前约 {temperature} 度，当前降水 {precipitation} mm。"
    return {
        "city": city,
        "date": date_value,
        "resolved_date": target_date,
        "forecast": forecast,
        "condition": condition,
        "temperature_c": temperature,
        "precipitation_mm": precipitation,
        "source": "live_open_meteo",
    }


def _resolve_city(city: str) -> tuple[float, float]:
    key = city.strip().lower()
    if city in CITY_COORDS:
        return CITY_COORDS[city]
    if key in CITY_COORDS:
        return CITY_COORDS[key]
    query = urllib.parse.urlencode({"name": city, "count": 1, "language": "zh", "format": "json"})
    url = f"https://geocoding-api.open-meteo.com/v1/search?{query}"
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"无法解析城市：{city}")
    return float(results[0]["latitude"]), float(results[0]["longitude"])


def _resolve_date(date_value: str) -> str:
    today = date_cls.today()
    value = (date_value or "today").strip().lower()
    if value in {"today", "今天"}:
        return today.isoformat()
    if value in {"tomorrow", "明天"}:
        return (today + timedelta(days=1)).isoformat()
    return value


def make_read_docs_tool(docs_dir: str | Path) -> ToolFunc:
    root = Path(docs_dir)

    def read_docs(arguments: dict[str, Any], session: Session) -> dict[str, Any]:
        query = arguments["query"].lower()
        matches: list[dict[str, str]] = []
        for path in root.glob("*.md"):
            text = path.read_text(encoding="utf-8")
            for paragraph in text.split("\n\n"):
                if query in paragraph.lower():
                    matches.append({"file": path.name, "snippet": paragraph.strip()[:600]})
        return {"query": arguments["query"], "matches": matches[:5]}

    return read_docs
