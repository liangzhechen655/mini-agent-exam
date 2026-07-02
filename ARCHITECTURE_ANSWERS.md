# Agent 架构设计题答案

这份答案的写法偏工程落地：先说明我理解的问题，再给方案，最后说明取舍。我的思路不是把所有名词都堆上去，而是站在一个要把 Agent 做成可用系统的人角度回答。

---

## 模块一：Context / Performance

### 1. 第一轮长窗口或多模态输入导致 first token 慢，怎么低成本从 5-10 秒压到 2 秒左右？

first token 慢，本质上是模型在输出前要先处理很长的输入。第一轮尤其明显，因为还没有缓存，system prompt、工具 schema、历史文档、图片内容都要一起 prefill。

如果目标是低成本、快速把用户体感从 5-10 秒压到 2 秒，我不会一上来改模型架构，而是先做几件工程优化。

第一，减少第一轮真正送给模型的内容。长文档不要整篇塞进去，先做目录、标题、摘要、关键词；多模态输入不要直接传高清原图或完整视频，先传缩略图、OCR 结果、关键帧和元数据。第一轮先让模型判断“需要看哪部分”，再按需补细节。

第二，把稳定前缀缓存起来。system prompt、工具 schema、产品说明、固定规则这些内容每轮基本不变，适合做 prompt caching。这样第二次以后请求可以复用前缀，不用每次重新计算。

第三，把“开始响应”和“完整完成”拆开。比如用户上传一个很长材料时，Agent 可以先在 1-2 秒内返回：“我先读取材料结构，随后重点分析第 2、3 部分。”与此同时后台继续解析全文。这不能减少全部计算量，但能明显改善用户等待体验。

第四，对多模态做渐进式加载。图片先低清，视频先关键帧，表格先 schema 和前几行样例。只有当模型判断需要精细内容时，再读取原始内容。

第五，使用流式输出。streaming 不一定减少真实 TTFT，但用户能更早看到系统开始工作，不会觉得页面卡死。

如果是我做产品，会把方案分成两层：

- 快速版本：前置摘要、裁剪输入、流式输出、后台解析。
- 进阶版本：prompt cache、chunked prefill、KV cache、专门的文档索引和多模态预处理。

一句话总结：第一轮慢不能只怪模型，很多时候是我们把太多不必要的东西一次性塞给它。低成本优化的重点是“先粗后细、先响应后补全、稳定内容缓存、长内容按需读取”。

### 2. 一个 session 连续聊了 200 轮，context 快爆了，怎么压缩？怎么保证流畅？

我会把 context 分成三层处理：最近消息、历史摘要、结构化状态。

第一层是最近消息。最近 10 到 20 条消息尽量原样保留，因为追问通常依赖最近几句话。比如用户问“那明天呢？”“第二个删掉”“刚才那个换成上海”，这些指代关系如果只靠摘要，很容易丢。

第二层是历史摘要。更早的消息不继续塞原文，而是压缩成 session summary。这个 summary 不是简单截断，而是保留后续对话真正需要的信息：

- 用户当前目标是什么
- 已经确认过哪些事实
- 工具查到过什么结果
- 有哪些未完成事项
- 用户有什么稳定偏好
- 最近讨论对象是谁，避免“它”“那个”断掉

第三层是结构化状态。凡是能结构化的东西，不应该只躺在自然语言里。例如 todo list、最近查询城市、任务进度、用户选择，都放到 session state。这样模型不用从一大段历史里猜状态。

我的项目里就是类似结构：

```text
messages: 最近对话和工具结果
summary: 被压缩的旧历史
state: todos、last_weather_city 等结构化状态
trace: 调试日志，不默认塞回 context
```

为了保证压缩后还流畅，我会注意三个点。

第一，压缩不要压掉最近上下文。刚发生的对话原样保留，旧内容才摘要。

第二，摘要要面向“继续对话”，不是面向“复述历史”。摘要里要写清楚用户目标、当前状态和下一步，而不是流水账。

第三，如果模型出现失忆，比如重复问已经说过的问题，可以从原始归档里按关键词或向量检索，把相关片段临时补回 context。

一句话总结：长 session 不应该靠无限扩大 context，而应该靠“最近窗口 + 语义摘要 + 结构化 state”来维持连续性。

---

## 模块二：Memory

### 1. 用户半个月后问以前问过的问题，Agent 怎么召回 memory？

我不会每次都把历史记忆全塞给模型。这样慢、贵，还容易把无关信息带进来。

更合理的流程是：

第一步，判断要不要召回。比如“今天上海天气”不需要长期 memory，调用天气工具即可；但“你还记得我之前那个笔试项目吗？”就明显需要 memory。

第二步，生成检索 query。用户经常说“上次那个”“之前说的”，这类句子本身信息量很低，所以 query 要结合当前 session 摘要一起扩展。

第三步，多路召回。至少分三类：

- session memory：当前窗口里的上下文
- episodic memory：过去某次对话的具体片段
- semantic memory：沉淀后的事实和偏好，比如“用户是初学者，喜欢详细解释”

第四步，排序和过滤。不能只看 embedding 相似度，还要看时间、重要性、用户是否明确要求记住、有没有新旧冲突。

第五步，带置信度注入 context。比如：

```text
可能相关记忆：用户之前在做 Agent 技术笔试，需要用小白能懂的方式解释。置信度：高。
```

如果涉及重要决策，我会让 Agent 确认，而不是装作百分百确定：

```text
我记得你之前倾向于用网页演示，不过那是之前的选择，现在还按这个来吗？
```

我认为 memory 的核心不是“记得越多越好”，而是“该想起来的时候能想起来，不该出现的时候不要污染当前对话”。

### 2. Agent memory 的经典框架和发展趋势是什么？

我理解的经典框架可以分四层。

第一层是 working memory，也就是当前 context window。模型当前能看到什么，就靠这一层。

第二层是 episodic memory，记录过去发生过的事件，比如某天讨论过一个方案、查过某个资料。

第三层是 semantic memory，记录比较稳定的事实和偏好，比如用户职业、语言偏好、长期项目。

第四层是 procedural memory，记录做事流程，比如每次提交代码前要跑测试、写 README、检查密钥是否泄露。

工程实现上，对应关系大概是：

- working memory：prompt、recent messages、state
- episodic memory：对话归档、向量检索
- semantic memory：用户画像、偏好表、知识图谱
- procedural memory：workflow、rule、skill、tool policy

发展趋势我觉得有三个。

第一，从“存聊天记录”变成“管理记忆生命周期”。记忆需要创建、更新、合并、遗忘，而不是只追加。

第二，从单纯向量库变成混合存储。向量适合语义相似，但不擅长表达关系和状态，所以会结合数据库、知识图谱和结构化 profile。

第三，从单一全局 memory 变成分层 memory。用户级、session 级、任务级要分开。比如“用户喜欢中文解释”是用户级；“window1 正在查上海天气”是 session 级；“这个部署任务跑到第 3 步”是任务级。

头部玩家大多也在往这个方向走：不只是把历史塞进长上下文，而是做 memory service、profile、retrieval、task state 和 tool trace 的组合。

---

## 模块三：Task

### 1. 长程任务里模型忘掉目标，有哪些解决方案？优缺点是什么？

长程任务里，模型不是故意忘，而是上下文里中间信息太多，原始目标被淹没了。

我知道的方案主要有几类。

第一，把目标每轮注入 system prompt。优点是简单有效，成本低；缺点是只知道目标还不够，模型还需要知道当前进度、已完成什么、下一步是什么。

第二，维护 task state。把目标、约束、步骤、当前进度、已完成事项、待办事项结构化存起来。优点是稳定，适合长任务；缺点是需要设计状态更新逻辑，不能完全靠模型自由发挥。

第三，做 checklist 或 plan。把大任务拆成多个小步骤，每完成一步就打勾。优点是可控；缺点是任务变化时要能动态调整 plan，否则会僵硬。

第四，checkpoint 自检。每隔几轮让 Agent 回答：当前动作是否还服务于目标？有没有偏离？下一步是什么？优点是能纠偏；缺点是多消耗 token 和时间。

第五，外部 supervisor。用另一个规则模块或模型监控当前 Agent 是否偏离。优点是适合高风险任务；缺点是系统复杂度更高。

我实际会组合使用：简单任务只用目标注入；中等任务用 task state + checklist；复杂任务再加 checkpoint 和 supervisor。

如果面试官问我“只把目标塞 system prompt 行不行”，我的回答是：这是必要步骤，但不充分。因为长程任务真正难的是进度管理和偏离纠正，不只是记住一句目标。

### 2. 用户要求每天早上 9 点根据昨天聊天做复盘总结，怎么设计？

这不是普通对话，而是一个定时激活任务。用户不发消息，Agent 也要到点自动醒来。

我会拆成五个模块。

第一，Task Registry。用户创建任务后，系统把它保存成结构化任务：

```json
{
  "task_id": "daily_review",
  "user_id": "A",
  "schedule": "09:00",
  "timezone": "Asia/Shanghai",
  "status": "active"
}
```

第二，Scheduler。每天早上 9 点扫描到期任务，创建一次 run_id，然后触发后台 Agent。

第三，Conversation Store。读取用户昨天 00:00 到 23:59 的聊天记录，包括消息、工具结果、todo 变化、关键决策。

第四，Summary Agent。它按固定模板生成复盘，比如：

- 昨天主要聊了什么
- 完成了什么
- 还有什么未完成
- 有哪些风险
- 今天建议先做什么

第五，Delivery。用户在线就发到聊天窗口，不在线就存成未读通知，也可以接企业微信或飞书。

还要考虑几个边界情况：昨天没有聊天就给空结果说明；聊天太多就先按 session 摘要再汇总；涉及 API key、身份证等敏感内容要脱敏；用户可以暂停、修改时间或删除任务。

一句话总结：这个需求需要 scheduler + conversation store + summary agent + delivery，不应该只靠当前聊天窗口里的 loop 硬做。

---

## 模块四：Tool / Session Runtime

### 1. 同步工具和异步工具怎么设计？

同步工具适合很快返回的任务，比如 calculator、weather、todo。流程是：模型决定调用工具，runtime 执行，结果马上回填给模型，模型继续判断或最终回答。

异步工具适合慢任务，比如部署服务、跑长测试、分析大量文件、生成视频。这类工具不能让用户一直等。

我的设计是把异步工具拆成两段。

第一段是 submit。runtime 生成 task_id、session_id、tool_call_id、run_id，把任务交给后台 worker，然后马上告诉用户：

```text
任务已开始，任务 ID 是 task_123，完成后我会通知你。
```

第二段是 complete。后台任务完成后发事件给 runtime：

```json
{
  "type": "async_tool_complete",
  "task_id": "task_123",
  "session_id": "A__window1",
  "tool_call_id": "call_abc",
  "result": {}
}
```

runtime 收到后，要把结果写回对应 session，再让模型生成用户能看懂的总结，最后通过网页消息、站内通知或企业微信通知用户。

关键点是：异步结果不能只弹一下就结束，它要进入 session，否则用户后面问“刚才那个结果展开说说”，Agent 就接不上。

### 2. session busy 时又来用户消息或异步工具结果，runtime 怎么处理？

原则是：同一个 session 内不要并发写状态。否则 messages、state、trace 很容易乱序或互相覆盖。

我会给每个 session 一个 event queue。

如果 session 是 idle，事件直接处理。

如果 session 是 busy：

- 普通用户消息：进入队列，提示用户“已收到，当前任务结束后继续处理”
- 取消消息：优先级最高，尝试中断当前 run
- 修正消息：如果还没到关键执行点，可以合并；否则排队到下一轮
- 异步工具完成事件：也进入队列，等当前 loop 到安全点再处理

为什么要有 run_id？因为异步结果可能过期。比如用户取消了一个部署任务，但部署系统几分钟后还是返回“成功”。这时 runtime 要检查 run_id 和任务状态。如果任务已经取消，就只写 trace，不要把它当成有效结果继续回复用户。

简单状态可以是：

```text
idle -> running -> waiting_tool -> running -> idle
                 -> cancelling -> idle
                 -> failed
```

我的笔试项目里实现了简化版 busy 状态：session 忙时先拒绝新消息。生产版本我会升级成 event queue。

---

## 模块五：Agent Runtime 架构对比

### 1. Claude Code 的工具输出方式和 GLM / 豆包等 OpenAI-compatible function calling 有什么不同？

Claude/Anthropic 的工具调用更像 content block。assistant 的 content 里可以出现文本块、tool_use 块；工具结果再以 tool_result 的形式放回消息里。它的特点是工具调用和文本更像同一条消息流里的不同块。

OpenAI-compatible function calling 更像结构化字段。模型返回 assistant message，其中 `tool_calls` 字段列出函数名和参数；应用执行工具后，再追加一条 `role=tool`、带 `tool_call_id` 的消息。

两者主要区别：

| 维度 | Claude 风格 | OpenAI-compatible 风格 |
|---|---|---|
| 工具调用位置 | content block 里 | `tool_calls` 字段里 |
| 工具结果 | `tool_result` block | `role=tool` 消息 |
| 参数形式 | 更像结构化对象 | 常见是 JSON 字符串 |
| 接入体验 | 表达力强 | 标准化、接入简单 |
| 适用场景 | coding agent、多步操作 | 普通业务工具调用、国内兼容模型 |

Claude 风格的优点是表达力强，适合 coding agent，因为模型可以在内容流中穿插工具调用。缺点是接入时要处理 content block 和流式事件，复杂一些。

OpenAI-compatible 的优点是生态广，国内 GLM、豆包、DeepSeek、Qwen 很多都能按类似方式接入。缺点是“兼容”不代表完全一致，不同厂商在 streaming、tool_choice、并行工具调用上可能有差异。

我的项目里做了一个兼容层：优先解析原生 `tool_calls`，如果模型只返回 JSON 文本，也能降级解析。这样 runtime 内部只关心统一后的 ParsedOutput。

### 2. OpenHands 的状态机设计有什么优缺？更优雅的实现方式是什么？

OpenHands 是偏 coding agent 的架构，核心是 action / observation / event loop。Agent 产生动作，比如读文件、写文件、执行命令；runtime 执行动作后产生 observation；这些事件不断回到 Agent，推动下一步。

它的优点很明显。

第一，可追踪。所有 action 和 observation 都有事件记录，方便回放和调试。

第二，职责分离。Agent 负责决定做什么，runtime 和 sandbox 负责真正执行，安全边界更清楚。

第三，适合 coding 场景。读文件、改代码、跑测试、看终端输出，都能抽象成 action。

但我觉得它也有几个问题。

第一，事件很多，长任务里 event log 会很长。虽然可追踪，但人要理解“当前任务到底做到哪了”并不轻松。

第二，全局状态机只能表达 running、paused、error 这类状态，表达不了任务语义。比如“正在修第 3 个测试失败点”这种信息通常藏在事件流里。

第三，错误恢复比较难。一个 command 失败，原因可能是依赖缺失、端口占用、代码冲突，不能只靠 retry。

我认为更优雅的方向是 event log + task graph + session queue。

event log 继续保留，用来审计和回放。

task graph 表达任务步骤，比如：

```text
理解需求 -> 读代码 -> 修改代码 -> 跑测试 -> 修复失败 -> 总结
```

每个节点都有状态：pending、running、done、failed。

session queue 处理并发。用户新消息、异步工具结果、定时任务都先进入队列，同一个 session 串行处理，避免状态乱掉。

也就是说，我不会只用一个大状态机描述全部东西，而是：

- 用 event log 记录发生过什么
- 用 task graph 记录任务做到哪
- 用 session state 记录当前稳定状态
- 用 queue 处理 busy 和异步事件

当然，对于笔试里的最小 Agent，不需要直接做成 OpenHands 那么复杂。我的项目用 `idle/busy` 加 trace 就够展示基础能力；如果要走生产级，再逐步引入 event queue、task graph 和 sandbox。
