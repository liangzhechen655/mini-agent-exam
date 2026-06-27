# 2026 Agent 技术笔试题：最小可用 Agent

这是一个从零实现的最小 Agent Runtime。项目没有使用 LangGraph、OpenHands、OpenClaw 等现成 Agent 框架，主循环、工具注册、工具调用、session 管理、context 压缩、trace 日志都在本项目内实现。

## 1. 运行环境

- Python 3.10+
- 真实 LLM API：支持 OpenAI-compatible `/v1/chat/completions`
- 本项目运行时只使用 Python 标准库

复制 `.env.example` 后设置环境变量：

```powershell
$env:OPENAI_API_KEY="你的 key"
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_MODEL="gpt-4o-mini"
```

如果使用其他兼容服务，例如公司网关或国内模型，只要提供兼容的 base url、key、model 即可。

也可以在项目根目录新建本地 `.env` 文件，程序会自动读取。`.env` 已被 `.gitignore` 忽略，不要提交到代码仓库：

```env
OPENAI_API_KEY=你的真实 key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

## 2. 启动方式

进入项目目录：

```powershell
cd D:\mini-agent-exam
```

单轮调用：

```powershell
py -m mini_agent.cli --user A --window window1 --once "帮我查上海今天的天气，并记一个待办：出门带伞"
```

交互式调用：

```powershell
py -m mini_agent.cli --user A --window window1
```

网页演示：

```powershell
chcp 65001
py -m mini_agent.web_app --host 127.0.0.1 --port 8765
```

然后打开：

```text
http://127.0.0.1:8765
```

网页和 CLI 复用同一个 `AgentRuntime`、同一套工具、同一个 session 存储。左侧可以切换窗口，中间是聊天，右侧可以查看 `state` 和 `trace`。

在 CLI 中：

- 输入自然语言即可聊天
- 输入 `/trace` 查看上一轮 LLM 决策和工具调用日志
- 输入 `/new` 创建新窗口
- 输入 `/switch window2` 切换窗口
- 输入 `/list` 查看当前用户所有窗口
- 输入 `/history` 查看当前窗口历史
- 输入 `/state` 查看当前窗口结构化状态
- 输入 `/exit` 退出

## 3. 测试方式

测试不依赖真实 API，而是用 `FakeLLM` 固定输出，验证 runtime 自身能力：

```powershell
py -m unittest discover -s tests -v
```

当前测试覆盖：

- LLM 输出 JSON 解析
- OpenAI-compatible 原生 `tool_calls` 解析
- `<tool_call>...</tool_call>` 文本降级解析
- 非法 JSON 重试恢复
- 工具参数校验失败
- 多步工具链：weather -> todo -> final
- 追问场景：基于上文继续调用工具
- calculator 工具
- calculator math 函数：`sqrt(16) + sin(pi / 2)`
- todo 工具状态变更
- weather/search/read_docs 工具，其中 weather 会返回 `source=live_open_meteo` 或 `source=mock_fallback`
- Agent 工具调用 loop
- trace 日志
- 同一用户不同窗口 session 隔离
- context 过长后的基础压缩

## 4. 系统设计

核心链路：

```text
用户输入
  -> 读取 user_id + window_id 对应 session
  -> 构造 system prompt、工具 schema、summary、recent messages、session state
  -> LLM 决策：优先原生 tool_calls，降级为 JSON 文本协议
  -> runtime 统一解析成 ParsedOutput
  -> 若 action=tool，校验参数并执行工具，把结果写入上下文，继续 loop
  -> 若 action=final，保存 assistant 回复并返回用户
```

关键文件：

- `mini_agent/runtime.py`：Agent 主循环、context 构造、工具执行、压缩
- `mini_agent/tools.py`：工具注册机制和默认工具
- `mini_agent/llm.py`：OpenAI-compatible LLM 客户端、原生 function calling 与文本协议解析
- `mini_agent/session_store.py`：JSON session 持久化
- `tests/`：单元测试

## 4.1 相比基础版本的增强点

这版吸收了另一个实现里的优点，但做了更稳的工程取舍：

- **双协议工具调用**：支持 OpenAI-compatible 原生 `tool_calls`，也保留手写 JSON 输出协议，模型或网关不支持 function calling 时仍可运行。
- **文本降级解析**：支持 `<tool_call>{...}</tool_call>`，提升对国产模型、弱 function calling 模型的兼容性。
- **正确工具消息协议**：工具结果进入上下文时使用 `role: "tool"` 和 `tool_call_id`，并保留对应 assistant `tool_calls`。
- **更强 CLI 演示**：支持 `/new`、`/switch`、`/list`、`/history`、`/state`、`/trace`，录屏时能直观看到多 session 隔离。
- **持久化 session**：不是只存在内存里，窗口关闭后仍可继续同一个 session。
- **Session 状态机预留**：session 包含 `idle/busy` 状态，方便后续扩展异步工具和事件队列。
- **LLM 重试**：模型输出不可解析时会追加格式修正提示并重试一次。
- **语义压缩**：真实运行时优先调用 LLM 生成历史摘要，失败时再用规则摘要兜底。
- **轻依赖**：运行时只依赖 Python 标准库，降低交付环境失败概率。
- **更多测试**：覆盖 JSON 协议、原生 tool_calls、文本降级解析、工具执行、session 隔离、context 压缩。

## 5. 工具注册机制

每个工具包含三部分：

- `name`：工具名
- `description`：给 LLM 的自然语言说明
- `parameters`：JSON Schema 风格参数描述

已实现工具：

- `calculator`：安全计算数学表达式
- `search`：mock 搜索
- `todo`：当前 session 内待办管理
- `weather`：真实天气查询，优先调用 Open-Meteo，失败时才 mock fallback
- `read_docs`：读取 docs 目录下的项目文档片段

本项目支持两种工具调用方式。

方式一：OpenAI-compatible 原生 `tool_calls`。runtime 会把工具转成：

```json
{
  "type": "function",
  "function": {
    "name": "weather",
    "description": "查询真实天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
  }
}
```

方式二：文本 JSON 协议。对于不稳定或不支持 function calling 的模型，LLM 可以输出：

```json
{"thought":"需要查天气","action":"tool","tool_call":{"name":"weather","arguments":{"city":"上海","date":"today"}}}
```

runtime 统一解析、校验、执行、记录日志，再把工具结果放回上下文。

## 6. Session 管理

session key 由 `user_id + window_id` 组成。

例子：

- 用户 A，窗口 1：`A__window1`
- 用户 A，窗口 2：`A__window2`

两个窗口分别持久化到不同 JSON 文件，因此待办、上下文、summary、trace 都互不影响。用户随时回到某个窗口继续聊，runtime 会读取对应 session。

## 7. Context 与 Memory 放置策略

放入 LLM context 的内容：

- system prompt：Agent 行为约束和 JSON 输出格式
- tool schema：让 LLM 自主选择工具
- session summary：被压缩的历史摘要
- session state：例如 todos、最近查询城市
- recent messages：最近用户输入、assistant 最终回复、工具结果

不反复放入 context 的内容：

- 详细 thought：只写 trace
- 全量历史消息：达到阈值后压缩
- 无关日志：只保存在 session trace 中

这么设计的原因：LLM 回复需要事实和状态，不需要每一步内部过程。thought 如果反复塞回上下文，会浪费 token，还可能污染后续决策。

## 8. Context 压缩策略

当前实现是基础压缩：

1. 当 `messages` 超过 `compress_threshold` 时触发
2. 保留最近 `max_recent_messages` 条消息
3. 把更早的 user、assistant、tool 消息抽取成 bullet summary
4. summary 最多保留 4000 字符
5. 压缩事件写入 trace

真实生产系统可以继续升级为：LLM 摘要、实体记忆抽取、任务状态机、向量召回、重要性评分。

## 9. 异常处理与 Trace

已处理的异常：

- LLM 输出不是合法 JSON
- 工具不存在
- 工具参数缺失或类型不匹配
- 工具执行报错
- 达到最大 loop 次数

每轮 trace 会记录：

- LLM 决策
- thought
- action
- tool name
- tool arguments
- tool result 或 error
- context compression 事件

CLI 中输入 `/trace` 可以查看上一轮 trace。

## 10. 录屏建议

录屏时建议演示 4 段：

1. 运行测试：`py -m unittest discover -s tests -v`
2. 启动网页：`py -m mini_agent.web_app --host 127.0.0.1 --port 8765`
3. 网页窗口 1：查询上海真实天气并添加待办
4. 网页新建窗口 2：添加另一个待办，再切回窗口 1 查看 state/trace

这样可以覆盖真实 LLM 调用、工具调用、多 session 隔离、追问、日志。

## 11. 代码链接

本地代码目录：`D:\mini-agent-exam`

提交时建议新建 GitHub/Gitee 仓库，然后执行：

```powershell
git init
git add .
git commit -m "implement minimal agent runtime"
git branch -M main
git remote add origin 你的仓库地址
git push -u origin main
```

最终在答卷里填写仓库链接即可。
