# AI Prompt 与问题解决记录

## 1. 任务理解 Prompt

用户原始任务是：完成 2026 年 Agent 技术笔试题，包括从零实现最小可用 Agent、测试用例、README、架构设计题答案，并面向初学者详细解释。

我把任务拆成两部分：

- Coding：实现一个不依赖现成 Agent 框架的 Agent Runtime。
- Writing：解释架构设计题，并把系统设计写清楚。

## 2. 关键设计 Prompt

我给自己的设计约束：

```text
做一个最小但完整的 Agent，不追求功能花哨。
必须体现：LLM 决策、工具 schema、工具执行、session 隔离、context 管理、压缩、trace、异常处理、测试。
代码要能用真实 LLM API 跑，但测试不能依赖真实 API。
```

## 3. LLM 输出协议 Prompt

放进 system prompt 的核心要求：

```text
你必须只输出一个 JSON 对象。
直接回答：
{"thought":"...","action":"final","final":"..."}
调用工具：
{"thought":"...","action":"tool","tool_call":{"name":"工具名","arguments":{}}}
```

这么做的原因：

- 不依赖 OpenAI 原生 function calling，便于兼容不同模型。
- runtime 可以自己实现解析逻辑，符合题目“LLM 输出解析逻辑”的要求。
- thought 可以记录 trace，但不必放回上下文。

## 4. 遇到的问题与解决

### 问题 1：真实 API 和测试稳定性冲突

如果测试直接调用真实 LLM，可能因为网络、额度、模型输出不稳定导致测试失败。

解决：实现 `OpenAICompatibleLLM` 用于真实运行；测试里实现 `FakeLLM`，固定输出 JSON，专门验证 runtime 行为。

### 问题 2：session 隔离如何直观体现

题目要求同一用户两个窗口互不影响。只用内存变量不够直观。

解决：用 `user_id + window_id` 生成 session_id，每个 session 保存到独立 JSON 文件。测试里让窗口 1 记录“查天气”，窗口 2 记录“写周报”，再回窗口 1 查询，验证状态不混。

### 问题 3：thought 要不要塞回上下文

如果每轮 thought 都塞回 context，会浪费 token，还可能让模型被自己的中间推理带偏。

解决：thought 只进入 trace，不进入 prompt messages。上下文只放用户输入、assistant 最终答复、工具结果、summary、session state。

### 问题 4：context 压缩复杂度

完整 memory 系统会很复杂，但笔试题要求基础压缩。

解决：实现一个轻量 extractive summary。超过阈值时保留最近消息，把旧消息抽成 bullet summary。todo 等结构化信息放 session state，不依赖摘要。

### 问题 5：工具参数校验

项目不引入第三方 jsonschema 依赖，但工具 schema 仍需要发挥作用。

解决：实现最小参数校验：required 字段和基础类型检查。生产环境可替换为完整 JSON Schema validator。

## 5. 后续可增强方向

- 接入原生 function calling，减少 JSON 解析失败。
- 加入异步工具队列和 run_id/tool_call_id。
- 加入长期 memory：事实抽取、向量召回、过期策略。
- 加入 Web UI，录屏更直观。
- 对工具结果做 token 预算裁剪，避免 search/read_docs 返回过长。
