# 架构设计题答案

说明：题面要求 5 个模块每个模块选一道题回答。下面是我选择的 5 道题，选择标准是尽量贴近本次最小 Agent Runtime 的实现，能把代码题和架构题串起来。

参考资料：

- Anthropic Tool Use: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- Anthropic How Tool Use Works: https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works
- OpenAI Function Calling: https://developers.openai.com/api/docs/guides/function-calling
- Z.AI GLM Function Calling: https://docs.z.ai/guides/capabilities/function-calling

## 模块一：Context / Performance

### 选题 2：一个 session 连续聊了 200 轮，context 快爆了。你会怎么做压缩？如何确保压缩后的对话仍然流畅？

我会把压缩分成三层：短期窗口、会话摘要、结构化状态。

第一层是短期窗口。最近几轮必须原样保留，因为它决定语言衔接、指代关系和追问体验。例如用户说“那明天呢”，如果只剩摘要，模型可能不知道“那”指天气、价格还是待办。所以最近 N 轮 user、assistant、关键 tool result 原样保留。

第二层是会话摘要。较早历史不直接丢掉，而是压缩成摘要。摘要不是文学总结，而是面向后续推理的工作记忆，应该包含：用户目标、已确认事实、重要偏好、已经完成的动作、未完成任务、关键工具结果、开放问题。摘要需要可迭代更新，每次压缩把旧 summary 和新旧消息一起合并，避免摘要越来越碎。

第三层是结构化状态。凡是可以结构化的东西，不应该只存在自然语言里。例如 todo list、订单状态、用户选择、任务进度、最近查询城市，都应该放 session state。这样模型不需要从一大段聊天里猜状态，runtime 可以稳定注入当前 state。

压缩流程可以这样做：

1. 达到 token 或轮次阈值后触发压缩。
2. 保留最近窗口，例如最近 10 到 20 条消息。
3. 对更早消息做摘要，工具结果只保留结论和影响，不保留冗长原文。
4. 从历史中抽取结构化 memory/state，例如待办、偏好、任务约束。
5. 压缩后做一次自检，让模型判断摘要是否能回答“用户现在在做什么、已经知道什么、下一步是什么”。

为了保证流畅，我会加两个机制。

一是摘要质量约束。摘要里必须保留指代锚点，例如“用户刚才查询的是上海天气”，这样后续“明天呢”能接上。二是失败恢复。如果模型回答时表现出断片，比如问了已经回答过的问题，runtime 可以触发更宽的历史回捞，从原始消息归档中找相关片段补回上下文。

本项目里的基础实现是：超过阈值后保留最近消息，把更早的 user、assistant、tool 结果抽成 bullet summary，并把 todo 等信息放在 session state。

## 模块二：Memory

### 选题 1：和聊天 Agent 熟悉半个月后，用户问了一个以前问过的问题。Agent 如何做 memory 召回更合理？

合理的 memory 召回不是“把半个月聊天全塞回去”，而是一个检索、筛选、合并、使用的过程。

第一步，判断是否需要召回。不是每句话都查长期记忆。比如“今天天气怎么样”更需要工具；“你还记得我上次问的论文方向吗”明显需要 memory；“继续上次那个方案”需要 session history 加 memory。

第二步，生成检索 query。query 不能只用用户原句，也要结合当前 session 摘要。例如用户问“之前那个接口怎么写”，query 应扩展成当前项目名、接口名、上次讨论主题。

第三步，多路召回。可以同时从三类 memory 里查：

- episodic memory：过去发生过的对话片段
- semantic memory：沉淀后的事实、偏好、知识
- task memory：未完成任务、计划、提醒、承诺

第四步，排序和过滤。排序维度包括语义相关度、时间新鲜度、用户显式确认程度、来源可靠性、隐私敏感性。半个月前的一句话不一定可靠，如果后来用户改过偏好，新记忆要覆盖旧记忆。

第五步，把 memory 放进上下文时要标注来源和置信度。例如“可能相关记忆：用户 6 月 12 日讨论过 XX，置信度中”。这样模型会把它当参考，而不是当绝对事实。

第六步，必要时向用户确认。尤其是涉及偏好、身份、承诺、钱、隐私时，不应该装作完全记得。更好的回答是：“我记得你之前提过 X，是继续按这个方向吗？”

我理解更合理的 memory 使用方式是：默认少召回，召回要精准；默认不暴露隐私，使用前要有必要性；默认承认不确定，而不是强行拟人化地说“我当然记得”。

## 模块三：Task / Reminder / Activation

### 选题 2：用户给 Agent 下达任务：每天早上 9 点根据昨天聊天情况做复盘总结。你会怎么设计？

这是一个定时激活任务，不应该依赖用户打开聊天窗口。系统需要把它从普通 session loop 升级成 scheduler + memory + notification 的组合。

核心模块：

- Task Registry：保存任务定义，例如用户、触发时间、时区、任务目标、通知渠道、权限。
- Scheduler：每天 9 点触发，必须用用户时区，避免跨时区错乱。
- Conversation Store：保存昨天的聊天记录和 session 摘要。
- Summary Agent：读取昨天相关内容，生成复盘。
- Delivery：把结果发到用户指定位置，例如聊天窗口、邮件、企业微信。
- Audit Log：记录任务何时触发、读取了哪些数据、是否成功发送。

执行流程：

1. 用户创建任务：“每天早上 9 点根据昨天聊天情况做复盘总结。”
2. Agent 追问或自动确认关键参数：时区、复盘范围、发送位置、是否包含敏感内容。
3. Task Registry 写入任务。
4. 每天 9 点 scheduler 产生 activation event。
5. runtime 根据 event 创建一次后台执行，不占用用户当前聊天 session。
6. Summary Agent 读取昨天 00:00 到 23:59 的会话摘要和关键原文。
7. 生成复盘：完成事项、未完成事项、重要决定、风险、今天建议。
8. 发送通知，并写 audit log。

要注意权限边界。用户说“昨天聊天情况”不等于所有应用数据都能读。系统应该只读取 Agent 负责的 conversation store，并支持用户指定“排除某些 session”或“不要包含私人内容”。

还要注意幂等。如果 9 点任务失败重试，不能重复发送三份总结。可以用 `task_id + date` 作为幂等键。

## 模块四：Tool / Session Runtime

### 选题 2：如果 session state 为 busy，此时用户又发来新消息，或者异步工具完成事件也到达，runtime 应该如何处理？

我会把 session 里的输入都事件化，然后用队列串行处理同一个 session 的状态变更。

核心原则：同一个 session 内，状态写入必须有顺序；不同 session 可以并行。

当 session busy 时来了用户新消息，有三种策略：

1. 排队：适合普通聊天。当前 loop 结束后再处理新消息。
2. 打断：适合用户说“停下”“取消”“不用做了”。需要向当前任务发送 cancel signal。
3. 合并：适合用户连续补充约束，例如“等等，城市改成上海”。runtime 可以把新消息作为补充事件注入当前计划。

策略选择不能只看 busy，还要看新消息意图。取消类消息优先级最高，补充类消息其次，普通追问排队。

异步工具完成事件到达时，也不要直接改 session。它应该进入同一个 session event queue。处理时检查：

- 这个 tool result 对应的 run_id 是否仍然有效
- 用户是否已经取消任务
- 当前 session 是否已经进入新的任务阶段
- 结果是否过期

如果结果仍有效，就把 tool result 写入 session，并触发一次 resume，让 Agent 根据结果继续。若结果过期，就写 trace，但不打扰用户。

我会给每次 Agent 执行分配 `run_id`，每个异步工具调用分配 `tool_call_id`。这样异步结果回来时可以精确关联，不会把窗口 1 的结果写进窗口 2，也不会把旧任务结果混进新任务。

状态机可以是：

- idle：可立即处理新消息
- running：正在 LLM loop 或同步工具
- waiting_tool：等待异步工具
- cancelling：正在取消
- failed：需要恢复或向用户解释

本项目是同步工具版本，但 trace 和 session_id 的设计可以自然扩展到异步事件队列。

## 模块五：Agent Runtime 架构对比

### 选题 1：Claude Code 的工具输出方式和国内 GLM / 豆包等 OpenAI-compatible function calling 有什么不同？他们各自这样设计的优缺点是什么？

我把这题理解为两类接口设计的对比：Claude/Claude Code 更偏“消息内容块里的工具使用协议”，OpenAI-compatible 模型更偏“chat completion 返回里的 tool_calls/function calling 字段”。具体产品会迭代，但核心差异是协议形态。

Claude 的工具使用通常是内容块协议。模型在 assistant message 中产生 `tool_use` 这样的结构化块，应用执行后再把 `tool_result` 放回对话。Anthropic 文档把它描述为应用和模型之间的契约：你声明工具和输入输出形状，模型决定何时请求工具，应用或服务器执行，结果再流回 conversation。

OpenAI-compatible function calling 通常是 `tools` + `tool_calls`。请求里提供 JSON Schema 工具定义，响应里返回 `tool_calls`，其中包含 `function.name` 和 JSON 字符串形式的 `function.arguments`。GLM 等兼容接口也采用类似字段，便于接入大量已有 OpenAI SDK 和 Agent 框架。

Claude 这类设计的优点：

- 工具调用和自然语言、文件、代码 diff、终端输出等都可以作为 conversation content 的一部分，适合复杂 Agent。
- 多步工具使用时，消息结构更接近真实交互轨迹。
- 对 Claude Code 这种 coding agent 来说，工具结果可以有更丰富的上下文形态。

Claude 这类设计的缺点：

- 客户端需要理解内容块协议，接入成本比单纯兼容 OpenAI 字段更高。
- 如果生态主要围绕 OpenAI-compatible SDK，适配层会更多。
- 流式工具输出、局部 JSON、长工具结果会增加 runtime 处理复杂度。

OpenAI-compatible function calling 的优点：

- 标准化程度高，很多模型、网关、框架都支持。
- 工具 schema、tool_calls、tool result 的工程链路清晰。
- 迁移成本低，适合快速把不同模型接到同一套业务系统。

OpenAI-compatible function calling 的缺点：

- 不同厂商虽然都说 compatible，但细节可能不同，例如参数是否是字符串、是否支持 parallel tool calls、是否严格校验 schema。
- 对复杂 coding agent 来说，单个 function call 抽象有时不够表达丰富环境变化。
- 如果模型只是“模板兼容”，实际工具调用稳定性可能弱于原生训练得更好的模型。

我的判断：如果做通用业务 Agent，OpenAI-compatible 更容易落地；如果做复杂 IDE/终端/文件系统 Agent，内容块式工具协议更自然。真正重要的不是选哪种格式，而是 runtime 要把工具调用视为可审计、可重放、可恢复的事件。
