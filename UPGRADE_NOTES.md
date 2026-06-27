# 升级说明：如何吸收 DS 版本优点并做得更稳

我审阅了 `C:\Users\30810\agent-exam` 的实现后，保留了其中值得学习的方向，并在 `D:\mini-agent-exam` 做了更适合提交的增强。

## 吸收的优点

1. **原生 function calling 思路**

DS 版本使用 OpenAI-compatible `tool_calls`，这是评阅者很容易认可的工程形态。新版已支持把本地工具 schema 转成 OpenAI tools 格式，并解析模型返回的 `tool_calls`。

2. **解析器分层**

DS 版本不只解析一种输出，而是有 fallback。新版保留原来的 JSON 协议，同时新增原生 `tool_calls` 和 `<tool_call>...</tool_call>` 文本降级解析。

3. **CLI 展示能力**

DS 版本有 `/new`、`/switch`、`/list`、`/history` 等命令，录屏展示更清楚。新版 CLI 已补齐这些命令，并新增 `/state`，能展示当前 session 的结构化待办、最近天气城市等状态。

4. **架构题表达更系统**

DS 版本的架构答案覆盖面更广。新版保留 `ARCHITECTURE_ANSWERS.md`，并在 README 里把代码设计与架构答案串起来，方便答辩时解释“代码如何对应设计”。

## 新版额外增强

1. **持久化 session**

DS 版本主要是内存 session。新版每个 session 存为独立 JSON 文件，用户关闭窗口后仍能继续聊，更贴近题目“随时接着窗口继续聊”的要求。

2. **运行时轻依赖**

新版不依赖 OpenAI SDK，只使用标准库 HTTP 调 OpenAI-compatible API。这样提交给别人运行时更少踩环境问题。

3. **工具状态在 Session 内**

todo 不使用全局 store，而是挂在当前 `session.state` 中。这样天然隔离窗口，也方便持久化和 trace。

4. **trace 不污染 context**

thought、工具参数、工具结果、错误都进 trace；只有必要的工具结果和结构化 state 进 context。这个设计比把所有中间日志塞给 LLM 更省 token，也更稳定。

5. **更多测试覆盖**

当前测试数从 7 个增加到 18 个，新增覆盖：

- 原生 OpenAI-compatible `tool_calls`
- `<tool_call>` 文本 fallback
- OpenAI tools schema 输出
- session store list
- 非法 JSON 后重试恢复
- 工具参数校验失败
- 多步工具链
- 追问场景
- weather/search/read_docs 工具
- calculator math 函数

6. **协议与运行稳定性修正**

新版把工具结果按 OpenAI-compatible 协议写回上下文：assistant 先记录 `tool_calls`，工具结果再以 `role: "tool"` 和 `tool_call_id` 返回。这样既能跑原生 function calling，也不会把工具结果伪装成用户消息。

7. **压缩和状态机更贴近生产**

压缩会在构造 prompt 前触发，避免本轮已经超长才事后压缩。真实运行时优先调用 LLM 做语义摘要，失败再规则兜底。Session 也增加了 `idle/busy` 状态，为异步工具和事件队列预留扩展点。

## 建议答辩话术

可以这样介绍：

> 我没有只做一个“能跑 demo”的 Agent，而是把 runtime 边界拆清楚了：LLM 只负责决策，runtime 负责解析、校验、执行、状态持久化、trace 和压缩。工具调用同时支持原生 function calling 和文本 JSON fallback，所以能适配 OpenAI、DeepSeek、GLM、豆包等不同兼容程度的模型。session 用 user_id + window_id 隔离并持久化，满足同一用户多个窗口互不影响、随时回来继续聊的要求。
