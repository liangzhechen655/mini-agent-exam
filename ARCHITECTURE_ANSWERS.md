# 架构设计题答案（全 10 题）

---

## 模块一：Context / Performance

### 题 1：大模型面对长窗口或多模态输入时，first token 会显著变慢。你有什么解决方案？

**根本原因**：Transformer 的 self-attention 计算量与序列长度平方成正比。长窗口（如 200K tokens）意味着每个 token 要对 200K 个 key 做 attention，prefill 阶段耗时巨大。多模态输入（图片、视频帧）token 化后往往产生极长序列（一张高清图轻松上千 token），进一步放大问题。

**解决思路：从硬件、模型架构、工程策略三个层面组合发力。**

**一、硬件 & 推理优化层**

1. **FlashAttention**：通过分块计算（tiling）和重计算（recomputation）把 attention 的 IO 复杂度从 O(N²) 降到 O(N)。这是当前最基础的加速手段，几乎所有推理引擎都已集成。

2. **KV-Cache 量化（KV-Quant）**：把 Key/Value 缓存从 FP16 压缩到 INT8 甚至 INT4，减少显存带宽压力。长序列场景下 KV cache 往往比模型权重还大，量化收益显著。

3. **Chunked Prefill**：不要一次性 prefill 整个 prompt，而是切成多个 chunk 分批处理。每个 chunk 的 prefill 和之前 chunk 的 decode 交替进行，平滑延迟峰值。

**二、模型架构层**

4. **GQA / MQA（Grouped Query Attention / Multi-Query Attention）**：减少 KV head 数量，直接降低 KV cache 大小和 attention 计算量。Llama 2/3、Mistral 等主流模型都已采用。

5. **稀疏 Attention / 滑动窗口**：不需要每个 token 关注所有历史 token。Sliding window attention（Mistral）、Longformer 的局部+全局模式，都能把有效计算量砍到 O(N)。

6. **多模态专用优化**：对图片做自适应分辨率（不要固定高分辨率）、早期视觉 token 压缩（如 LLaVA 的 spatial pooling）、动态帧率（视频不抽 30fps，抽关键帧）。

**三、工程策略层**

7. **Prompt Caching（最重要、最实用）**：把 system prompt、工具 schema、历史消息等不变前缀缓存起来。Anthropic API 支持显式 cache breakpoint，第二次请求相同前缀时 prefill 直接跳过，TTFT 可降低 80%+。本项目 `_build_prompt_messages` 里 system prompt + tool schema 永远不变，非常适合做 cache breakpoint。

8. **前缀压缩 / 提前摘要**：不要等到 context 爆了才压。在发送前主动把历史消息压缩成摘要，用 300 字摘要替代 3000 字原文。

9. **渐进式加载多模态内容**：先发低分辨率缩略图让模型快速判断是否需要细节，确认后再发高分辨率原图。视频先发首帧 + 元数据。

10. **Streaming 掩盖延迟**：虽然不能减少 first token 时间，但让用户看到 token 逐个出现，感知延迟大幅降低。

**一句话总结**：Prompt Caching + Chunked Prefill + GQA 是当前最经济实用的三板斧。FlashAttention 是基础设施，应该默认开启。

---

### 题 2：一个 session 连续聊了 200 轮，context 快爆了。你会怎么做压缩？如何确保压缩后的对话仍然流畅？

（本题已在初版作答，这里做更深入的工程细节补充）

**三层压缩架构**（已在 `mini_agent/runtime.py` 中实现原型）：

**第一层：保留窗口（Sliding Window）**

最近 N 条消息原样不动。N 的选择是关键：
- 太小（<6）：追问"那明天呢？"可能丢失上文消息，模型找不到指代锚点
- 太大（>20）：压缩意义减弱
- 经验值 10-16 条，取决于对话密度

边界保护：如果窗口起始恰好是一条 tool 消息，向前多保留一条对应的 assistant 消息。否则 LLM 会看到孤立的 tool result 却不知道谁调用的（`_recent_messages` 第 253-260 行的实现）。

**第二层：语义摘要（Semantic Summary）**

不是简单的文本截断，而是面向后续推理的"工作记忆摘要"。必须保留的内容：
- 用户目标（为什么开启这段对话）
- 关键事实（查过的数据、执行过的工具结果）
- 未完成任务（待办、进行中的事情）
- 指代锚点（"刚才讨论的是上海天气"，保证"明天呢"能接上）
- 用户偏好（如果用户说过"我不喜欢缩写"，后续回答应记住）

迭代合并原则：旧摘要 + 新压缩的摘要合并，避免摘要碎片化。总长度限制（如 4000 字符）。

本项目的 LLM 优先 + 规则兜底策略是务实的：LLM 能理解语义但不可靠，规则 bullet list 一定能跑通。

**第三层：结构化状态（Structured State）**

凡是能从对话中提取为结构化数据的，就不该只躺在自然语言里：

| 信息类型 | 存储形式 | 示例 |
|---------|---------|------|
| 待办列表 | `state["todos"]` | `[{id:1, text:"出门带伞", done:false}]` |
| 最近查询 | `state["last_weather_city"]` | `"上海"` |
| 用户偏好 | `state["preferences"]` | `{language:"zh", detail_level:"brief"}` |
| 任务进度 | `state["task_progress"]` | `{step:3, total:7, current:"review PR"}` |

**保证流畅的两个额外机制**：

1. **压缩自检**：压缩后让模型判断"从摘要能否回答：用户在做什么、知道了什么、下一步要做什么"。不合格则降低压缩力度（保留更多消息）。

2. **断片恢复**：如果模型回答时表现出失忆（重复问问题、答非所问），runtime 自动触发历史回捞——从归档的原始消息中按语义检索相关片段，临时补回 context。

---

## 模块二：Memory

### 题 1：和聊天 Agent 熟悉半个月后，用户问了一个以前问过的问题。Agent 如何做 memory 召回更合理？

**核心原则：精准召回 > 全量注入；承认不确定 > 强行拟人。**

**六步召回流程**：

**Step 1 — 判断是否需要召回**：不是每句话都查 memory。"今天天气怎么样" → 调工具即可；"你还记得我上次问的论文方向吗" → 明显需要 memory；"继续上次那个方案" → 需要 session history + memory。

**Step 2 — 生成检索 query**：只用用户原句不够。应结合当前 session 摘要做 query 扩展。例如用户问"之前那个接口怎么写"，query 应扩展成 `[当前项目名] [接口名] [上次讨论主题]`。

**Step 3 — 多路召回**：从三类 memory 同时查：
- **Episodic Memory**：过去对话的具体片段。"6 月 12 日用户讨论过 XX 论文的实验设计"
- **Semantic Memory**：沉淀后的事实和偏好。"用户偏好简洁回答，不需要背景知识解释"
- **Task Memory**：未完成任务、承诺、提醒。"上周用户说过'下周一提醒我交报告'"

**Step 4 — 排序与过滤**：多维度打分：
- 语义相关度（embedding similarity）
- 时间衰减（越近权重越高，但不线性——昨天的重要决定 > 一小时前的闲聊）
- 用户显式确认程度（用户说过"记住这个"的加权）
- 冲突检测（如果后来用户改过偏好，新 > 旧）
- 隐私敏感度（涉及身份、密码的记忆需要降级处理）

**Step 5 — 标注置信度注入 context**：不要说"我知道你之前 X"，而应在 context 里标注："[可能相关记忆，置信度中] 用户 6 月 12 日讨论过 XX 论文方向"。让模型把它当参考而非事实。

**Step 6 — 必要时向用户确认**：涉及偏好变更、重要承诺、金钱、隐私时，更好的回答是："我记得你之前提过 X，不过那已经是两周前了，还是按这个方向吗？"

**为什么不是"把所有记忆 dump 进 context"**：
- 200 轮聊天 × 半个月 = 数千条消息，全塞进去 LLM 会淹没在噪声里
- 隐私风险：用户可能聊过很多不同话题，不该让 LLM 看到所有
- 成本：context 越长越贵越慢

---

### 题 2：你理解的 Agent memory 经典框架是什么？它的发展趋势是什么？

**经典框架：类人认知架构（Cognitive Architecture）**

学术界和工业界最常引用的框架源自认知心理学，把 Agent memory 分为四个层次：

```
┌──────────────────────────────────────────────────┐
│                Agent Memory 四层模型               │
│                                                  │
│  ┌──────────┐  最短（秒-分钟），容量极小           │
│  │ Working  │  当前 context window               │
│  │ Memory   │  就是 LLM 能直接看到的 token         │
│  └────┬─────┘                                    │
│       │ 关注的信息写入                            │
│  ┌────▼─────┐  中短期（小时-天），容量中等         │
│  │ Episodic │  对话片段、事件、经验                │
│  │ Memory   │  "上次用户说过..."                   │
│  └────┬─────┘                                    │
│       │ 反复出现、被确认的沉淀                    │
│  ┌────▼─────┐  长期（周到月），容量大              │
│  │ Semantic │  事实、偏好、知识                    │
│  │ Memory   │  "用户是后端工程师，喜欢简洁回答"     │
│  └────┬─────┘                                    │
│       │ 技能化、自动化                            │
│  ┌────▼─────┐  永久，容量无限                      │
│  │Procedural│  技能、流程、习惯                    │
│  │ Memory   │  "每次代码 review 前跑一遍 lint"     │
│  └──────────┘                                    │
└──────────────────────────────────────────────────┘
```

**各层对应的工程实现**：

| 层次 | 工程实现 | 代表工具/框架 |
|------|---------|-------------|
| Working Memory | Context Window（system prompt + messages + state） | 所有 LLM |
| Episodic Memory | 向量数据库（Chroma, Pinecone, Milvus） | MemGPT, Mem0, LangChain Memory |
| Semantic Memory | 知识图谱（Neo4j）+ 向量检索 + 结构化存储 | Letta, Cognee |
| Procedural Memory | Rule engine, Workflow, Skill 注册表 | Claude Code Hooks, LangGraph |

**代表框架**：

1. **MemGPT / Letta**（2023，伯克利）：最早提出"给 LLM 虚拟内存管理"的概念。把 context window 视为虚拟内存，对话历史按页换入换出。LLM 自己通过 function calling 决定何时从长期存储"翻页"。核心洞察：LLM 的 context 就是它的 working memory，memory 管理的本质是 context 的分页调度。

2. **Mem0**（2024）：把 memory 抽象为增删改查操作。自动从对话中提取偏好、事实、决策，存为独立 memory record。支持去重、更新（新偏好覆盖旧偏好）、衰减。设计上更偏向 user profile 记忆。

3. **LangChain Memory**：ConversationBufferMemory、SummaryMemory、VectorStoreRetrieverMemory 等。优点是生态成熟，缺点是抽象层太多，memory 更新逻辑不够智能，更像把数据存储包装成 memory。

4. **Cognee**：用图数据库构建 knowledge graph，把对话中的实体和关系结构化。memory 不仅是文本片段，还是可查询的图。

**发展趋势**：

1. **从被动存储到主动管理**：早期 memory = 存起来以后查。现在趋势是 Agent 主动决定什么该记、什么该忘、什么该更新。记忆有自己的生命周期（create → update → decay → delete）。

2. **从单一向量检索到多模态图记忆**：纯 embedding 检索会漏掉结构化关系（"张三的老板是李四"）。GraphRAG 把知识图谱和向量检索融合，记忆不仅是片段，更是可推理的实体关系网。

3. **从全局记忆到分层记忆**：不同 scope 的记忆分开管理——user-level（跨 session）、session-level（当前对话窗口内）、task-level（当前任务上下文）。本项目 `state` dict + `summary` + `messages` 三层结构就是最简单的分层实现。

4. **隐私与遗忘权**：GDPR 和各国数据法规推动 "machine unlearning"。memory 系统需要能精确删除某个用户的所有记忆，而不是模糊地从向量库里删近似向量。

5. **Memory-as-a-Service**：Mem0、Letta 都在走向独立的 memory service，不绑定特定 Agent 框架。Agent runtime 调用 memory API 就像调用数据库。

**一句话总结**：Agent memory 正在从"一个大 context window"走向"分层、可管理、可更新的认知架构"。MemGPT 的核心洞察——context 就是 working memory，memory 管理的本质是调度——是理解这个领域的钥匙。

---

## 模块三：Task / Reminder / Activation

### 题 1：对于长程任务，大模型执行一段时间可能会忘掉目标。如何用 reminder 或其他机制保证任务稳定执行？

**问题本质**：LLM 无状态——每次推理只看当前 context。长程任务执行几十轮后，早期目标可能被压缩掉、或淹没在大量中间结果里。模型开始"跑偏"——做和原始目标无关的事。

**解决策略：目标注入 + 进度追踪 + 自动纠偏。**

**一、目标持久化与周期性注入**

不要把目标只写在第一条 user message 里。要结构化存储并周期性地重新注入 context：

```
每次 LLM 决策的 system prompt 里强制包含：
┌────────────────────────────────────────┐
│ 🎯 当前任务目标：分析上周的销售数据，  │
│    找出下降最多的品类并给出建议。       │
│ 📍 当前进度：步骤 2/4 — 正在分析数据    │
│ ⏳ 已完成：1.获取数据 ✓                │
│ 🔜 待完成：3.生成图表  4.写报告         │
│ ⚠️ 约束：不能修改原始数据、报告不超过2页 │
└────────────────────────────────────────┘
```

工程实现上类似本项目的 `session.state` + system prompt 注入：`runtime.py:226-227` 把 state 作为 system prompt 的一部分。对于长程任务，state 里应包含 `current_task_goal`、`task_steps`、`completed_steps`。

**二、检查点与子目标分解**

长程任务拆成多个子目标，每个子目标结束后自检：

```python
# 每个子目标完成后
checkpoint_prompt = f"""
原始目标: {original_goal}
已完成: {completed_steps}
当前产出: {current_output}
请判断：当前产出是否仍在为实现原始目标服务？
如果偏离，请纠正方向。
"""
```

如果自检发现偏离，自动回滚到上一个检查点重新执行，而不是继续在错误方向上浪费 token。

**三、进度条机制（Progress Bar）**

把模糊的"做一个分析报告"变成可度量的进度：

```
Phase 1: 获取数据        [████████████] 100%
Phase 2: 数据清洗        [████████████] 100%
Phase 3: 分析建模        [██████░░░░░░]  50% ← 当前
Phase 4: 生成报告        [░░░░░░░░░░░░]   0%
```

进度条有两个作用：一是让模型知道自己做到哪了；二是当进度条不动时，runtime 可以触发告警。

**四、自动纠偏触发器**

设置偏离检测规则：

| 触发条件 | 纠偏动作 |
|---------|---------|
| 连续 3 轮没产出和任务目标相关的工具调用 | 注入 reminder："注意，当前任务是 XX" |
| 模型输出了和目标无关的 final answer | 不发送给用户，而是让模型重新规划 |
| 某步骤结果和预期严重不符 | 标记 step 为 failed，尝试替代方案 |
| 执行时间超过预期 2 倍 | 暂停并询问用户是否继续 |

**五、Reminder 队列**

对于需要跨越多个 session 的任务（如"下周交报告"），不在 context 里死等。而是在 scheduler 里注册 reminder：

```
Task: 提醒用户提交季度报告
Reminder: 每天检查一次，截止前 3 天开始每天提醒
Activation: 下次用户打开聊天时，或在截止日期当天主动推送
```

**为什么不能用"把目标每轮都放 system prompt 里"一个方法解决**：那只是消极防御。真正有效的机制是：目标持久化 + 进度可视化 + 自检纠偏 + 必要时外部触发提醒。四层防护。

---

### 题 2：用户给 Agent 下达任务：每天早上 9 点根据昨天聊天情况做复盘总结。你会怎么设计？

（本题已在初版作答，这里补充更多工程实现细节）

**核心设计：这不是一个 session loop 能解决的。需要 Scheduler + Task Registry + Conversation Store + Summary Agent + Delivery 五个模块协作。**

**架构图**：

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌──────────┐   每天9点触发     ┌──────────────────┐    │
│  │ Scheduler │ ──────────────→ │ Task Dispatcher   │    │
│  │ (cron)   │                 │ 生成 activation   │    │
│  └──────────┘                 └────────┬─────────┘    │
│                                        │               │
│                          创建独立执行上下文              │
│                                        │               │
│  ┌─────────────────────────────────────▼──────────┐    │
│  │           Background Execution Context         │    │
│  │                                                │    │
│  │  ┌───────────┐  读取昨天的    ┌──────────────┐ │    │
│  │  │ Summary   │ ←─────────── │ Conversation │ │    │
│  │  │ Agent     │   conversations│ Store       │ │    │
│  │  │ (LLM)     │               │ (昨天0-24时)  │ │    │
│  │  └─────┬─────┘               └──────────────┘ │    │
│  │        │ 生成复盘                             │    │
│  │        ▼                                      │    │
│  │  ┌───────────┐                               │    │
│  │  │ 复盘结果   │                               │    │
│  │  │ - 完成事项 │                               │    │
│  │  │ - 未完成   │                               │    │
│  │  │ - 关键决定 │                               │    │
│  │  │ - 风险提示 │                               │    │
│  │  │ - 今日建议 │                               │    │
│  │  └─────┬─────┘                               │    │
│  └────────┼──────────────────────────────────────┘    │
│           │                                            │
│           ▼                                            │
│  ┌──────────────┐     ┌──────────────┐                │
│  │ Delivery     │     │ Audit Log    │                │
│  │ 聊天窗口/邮件 │     │ 执行记录      │                │
│  └──────────────┘     └──────────────┘                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**关键设计决策**：

1. **时区感知**："早上 9 点"必须是用户的本地时间，不是服务器 UTC。Task Registry 存储 `trigger_time: "09:00"` + `timezone: "Asia/Shanghai"`。夏令时变更时自动调整。

2. **不占用用户 session**：复盘任务在独立 context 里执行，不和用户当前聊天窗口的 messages 混在一起。但复盘结果可以投递到用户聊天窗口。

3. **读取权限边界**："昨天聊天情况" ≠ Agent 的所有数据。只读取用户授权范围内的 conversation store。用户可指定排除特定敏感 session。

4. **幂等性**：Scheduler 可能因为重启、网络抖动重复触发。用 `task_id + date(YYYY-MM-DD)` 作为幂等键。当天已经生成过复盘 → 跳过。

5. **失败处理**：如果复盘生成失败（LLM 挂了、数据不可读），记录错误日志，降级为简单统计（对话轮数、工具调用次数），并通知用户"今天复盘因 XX 原因不完整"。

6. **复盘 Prompt 设计**：

```
你是一个每日复盘助手。请根据以下昨天的对话摘要生成复盘。

要求：
1. 完成事项：列出昨天明确完成的任务和决定
2. 未完成事项：正在推进但未完成的事项
3. 关键决定：重要的决策和偏好变更
4. 风险提示：可能被忽略但重要的事项
5. 今日建议：基于昨天情况，给出 1-3 条今天的行动建议

格式：简洁、可操作，不要客套话。总字数不超过 500 字。
```

---

## 模块四：Tool / Session Runtime

### 题 1：Agent 工具有同步和异步两类。异步工具不能让用户一直等，但结果依然重要。你会如何设计异步工具执行和完成通知？

**设计目标**：同步工具快速返回、异步工具不阻塞主 loop、结果回到对话时上下文完整。

**一、异步工具的生命周期**

```
  ┌─────────┐   submit    ┌──────────┐   poll/callback  ┌──────────┐
  │ Agent   │ ──────────→ │ Async    │ ───────────────→ │ Agent    │
  │ Runtime │            │ Executor │                 │ Runtime  │
  │         │ ←─ 立刻返回  │          │  ← 完成通知       │          │
  │         │   pending   │ (后台跑)  │   + result       │          │
  └─────────┘            └──────────┘                 └─────────┘
       │                      │                            │
       │ 告诉用户：            │                            │ 把结果写入
       │ "正在处理，           │                            │ session，
       │  稍后通知你"          │                            │ 触发resume
```

**二、工具注册时声明同步/异步**

```python
# 工具注册时增加 mode 字段
tools.register(
    name="deploy_to_k8s",      # 异步：部署可能需要 5 分钟
    description="部署应用到 K8s 集群",
    parameters={...},
    mode="async",               # ← 关键：声明为异步
    timeout_seconds=600,
)

tools.register(
    name="calculator",          # 同步：毫秒级返回
    description="计算数学表达式",
    parameters={...},
    mode="sync",
)
```

**三、异步执行流程**

```python
def _execute_tool(self, session, name, arguments, step, tool_call_id):
    tool_spec = self.tools.get(name)

    if tool_spec.mode == "sync":
        # 同步：当场执行，当场返回
        result = self.tools.execute(name, arguments, session)
        return {"ok": True, "tool": name, "result": result}

    elif tool_spec.mode == "async":
        # 异步：提交任务，立刻返回 pending 状态
        task_id = self.async_executor.submit(
            tool_name=name,
            arguments=arguments,
            tool_call_id=tool_call_id,
            session_id=session.session_id,
        )
        # 告知 LLM 工具已提交，不要空等
        return {
            "ok": True,
            "tool": name,
            "status": "pending",
            "task_id": task_id,
            "message": f"任务已提交，预计 {tool_spec.estimated_time}。结果稍后通知。"
        }
```

**四、结果回写与通知机制**

异步任务完成后，结果如何回到对话？两种互补方式：

**方式 A：轮询（适合短异步，<30 秒）**

```python
# Agent 在下一轮 loop 中主动检查
for pending_call in session.pending_async_calls:
    status = self.async_executor.check(pending_call.task_id)
    if status.done:
        # 结果写回 messages，就像同步工具返回一样
        session.add_message("tool", json.dumps(status.result),
                           name=pending_call.name,
                           tool_call_id=pending_call.tool_call_id)
```

**方式 B：事件推送（适合长异步，分钟到小时）**

```python
# 异步执行器完成后 push 事件到 session event queue
async_executor.on_complete = lambda task_id, result:
    session_event_queue.push({
        "type": "async_tool_complete",
        "session_id": session_id,
        "tool_call_id": tool_call_id,
        "result": result,
        "completed_at": now_iso(),
    })
```

**五、用户侧体验**

```
用户: 帮我把这个服务部署到 K8s

Agent: 正在为你部署到 K8s 集群...（任务ID: dep-2024）
      预计需要 3-5 分钟，完成后我会通知你。

[3分钟后]

Agent: ✅ 部署完成！服务已上线，访问地址 https://xxx.com
      Pod 状态: Running，副本数: 3
```

**关键设计**：
- 异步工具返回的 pending 状态也要写入 messages，LLM 下次决策时知道有任务在跑
- `tool_call_id` 贯穿始终——提交时生成，返回时精确匹配，不会把部署结果错接到计算器调用上
- 用户可以中途取消：`session.add_message("user", "取消部署 dep-2024")` → runtime 调用 `async_executor.cancel(task_id)`

---

### 题 2：如果 session state 为 busy，此时用户又发来新消息，或者异步工具完成事件也到达，runtime 应该如何处理？

（本题已在初版作答，这里补充更详细的状态机设计和实现伪代码）

**核心设计：输入事件化 + 单 session 串行队列 + 多 session 并行。**

**完整状态机**：

```
                    ┌──────────┐
           ┌──────→│   IDLE   │←──────────┐
           │       └────┬─────┘           │
           │            │                  │
           │     用户发消息                 │ 任务完成 /
           │            │                  │ 取消确认
           │            ▼                  │
           │       ┌──────────┐    异步工具 │
           │       │ RUNNING  │────调用────→│
           │       └────┬─────┘            │
           │            │                  │
           │     用户说"停止"               │
           │            │                  │
           │            ▼                  │
           │       ┌──────────┐           │
           └───────│CANCELLING│───────────┘
                   └────┬─────┘
                        │
           取消完成 / 超时
                        │
                        ▼
                   ┌──────────┐    手动修复后
                   │  FAILED  │──────────────→ IDLE
                   └──────────┘
```

**状态定义**：

| 状态 | 含义 | 允许的输入 |
|------|------|----------|
| `idle` | 空闲，可立即处理 | 用户消息、定时任务、异步结果 |
| `running` | 正在 LLM loop 中 | 取消命令、异步结果（排队） |
| `waiting_tool` | 等待异步工具 | 异步结果、取消命令、用户消息（排队或合并） |
| `cancelling` | 正在取消当前任务 | 无（等待取消完成） |
| `failed` | 异常终止 | 用户重试、手动恢复 |

**三种新消息处理策略的选择逻辑**：

```python
def handle_incoming_during_busy(session, incoming_event):
    if session.status == "idle":
        return "process_immediately"

    if incoming_event.type == "user_message":
        intent = classify_intent(incoming_event.content)

        if intent == "cancel":       # "停下" "取消" "不用了"
            return "interrupt"        # 发送 cancel signal 给当前任务

        elif intent == "amend":       # "等等，城市改成上海"
            return "merge"            # 作为补充注入当前 context

        else:                         # 普通追问
            return "enqueue"          # 排队，等当前 loop 结束

    if incoming_event.type == "async_tool_complete":
        # 检查结果是否还有效
        if incoming_event.run_id != session.current_run_id:
            return "stale"            # 来自已取消的任务，只写 trace
        if session.status in ("cancelling", "failed"):
            return "stale"
        return "resume"               # 把结果写入 messages，重新进入 loop
```

**Run ID 机制**：每次 `chat()` 调用生成一个新的 `run_id`。异步工具调用带上 `run_id`。结果回来时对比——如果 `run_id` 不匹配，说明用户已经在另一个 `chat()` 里开始了新对话，旧工具结果不应该混入当前 context。

**Session Event Queue（扩展方向）**：

当前简单实现是 busy 时直接拒绝新消息（`runtime.py:59-64`）。更完善的方案是引入 event queue：

```python
# 将当前项目从"拒绝"升级为"排队"
if session.status == "busy":
    session.event_queue.append({
        "type": "user_message",
        "content": user_input,
        "timestamp": now_iso(),
    })
    return AgentResponse(
        answer="当前正在处理中，你的消息已排队，处理完后会立即继续。",
        status="queued",
    )

# 当前 loop 结束后，检查 queue
# finally:
#     if session.event_queue:
#         next_event = session.event_queue.pop(0)
#         self.chat(user_id, window_id, next_event.content)
```

---

## 模块五：Agent Runtime 架构对比

### 题 1：Claude Code 的工具输出方式和国内 GLM / 豆包等 OpenAI-compatible function calling 有什么不同？他们各自这样设计的优缺点是什么？

（本题已在初版作答，这里补充本项目的对照和更深入的分析）

**两种协议的本质差异**：

| 维度 | Anthropic Tool Use | OpenAI-compatible Function Calling |
|------|-------------------|-----------------------------------|
| **工具调用位置** | 在 `content` 流里，作为内容块（content block） | 在 `tool_calls` 字段里，独立于 `content` |
| **类型标识** | `"type": "tool_use"` | `"role": "assistant"` + `"tool_calls": [...]` |
| **参数格式** | JSON object（结构化） | JSON string（需二次 parse） |
| **结果回写** | `"type": "tool_result"` 内容块 | `"role": "tool"` 消息 + `"tool_call_id"` |
| **与文本关系** | 文本和工具调用可穿插（交错） | 文本在 `content`，工具在 `tool_calls`，分离 |
| **并行调用** | 多个 `tool_use` 块 | `tool_calls` 数组，支持并行 |
| **流式处理** | content block 事件流 | delta 累积拼接 |
| **缓存** | 支持 cache breakpoint 标注 | 部分厂商支持（需查文档） |

**Claude 设计哲学**：工具调用是"对话的一部分"。模型在一个 assistant turn 里可以先说一段话，然后调用工具，再接着说。这对 coding agent 特别重要——"我先检查文件结构（查阅），再修改代码（编辑），然后解释我改了什么（文本）"都在同一条消息里。

**OpenAI-compatible 设计哲学**：工具调用是"模型的结构化输出"。`content` 给你看的文本，`tool_calls` 给你的代码执行的指令。两者分离，方便程序化处理——你的代码不需要从自然语言里 parse 出 function call。

**在本项目中的体现**（`llm.py`）：

本项目同时支持两种协议。`parse_llm_message()` 优先解析原生 `tool_calls` 字段，如果 LLM 不支持（只返回纯文本），则降级到 `parse_llm_output()` 从 JSON 文本中解析工具调用。

**优缺点深化**：

Claude 内容块式：
- 优点：表达力强，适合多步推理+coding、模型可在文本中间精确放置工具调用
- 缺点：接入复杂度高，需要处理流式内容块事件，生态工具较少

OpenAI-compatible：
- 优点：标准化，接入简单（JSON Schema → tool_calls → tool result），生态成熟
- 缺点：`arguments` 是 JSON 字符串而非对象（额外 parse）、厂商兼容性参差不齐、对复杂 coding agent 表达力不足

**另一个本项目的关键设计**：无论哪种协议，进入 session.messages 后统一用 OpenAI 格式存储（`_to_openai_tool_call` 做标准化转换）。这样 context 构建、压缩、重放都只需要处理一种格式。这个"内部统一格式"的思路可以在更复杂的 runtime 里复用。

---

### 题 2：OpenHands 的状态机设计有什么优缺？更优雅的实现方式是怎么样的？

**OpenHands（原 OpenDevin）架构概述**：

OpenHands 是一个 AI 编码 Agent，核心架构是事件驱动的 Agent-Action-Observation 循环。状态机围绕 `AgentState` 枚举展开：

```
         ┌──────────┐
    ┌───→│  LOADING │  (初始化 agent 配置)
    │    └────┬─────┘
    │         ▼
    │    ┌──────────┐
    │    │  INIT    │  (设置环境、克隆仓库)
    │    └────┬─────┘
    │         ▼
    │    ┌──────────┐
    │    │ RUNNING  │←──────┐
    │    └────┬─────┘       │
    │         │             │
    │    ┌────┼─────────┐   │
    │    ▼    ▼         ▼   │
    │  AWAITING_USER  PAUSED│
    │    │    │         │   │
    │    │    └────┬────┘   │
    │    │         ▼        │
    │    │    ┌──────────┐  │
    │    └───→│  ERROR   │  │
    │         └────┬─────┘  │
    │              │         │
    │         ┌────▼─────┐   │
    │         │ FINISHED │   │
    │         └──────────┘   │
    │                        │
    └──── 状态转换通过 Event ─┘
```

**OpenHands 状态机的优点**：

1. **事件驱动（Event Sourcing）**：每个 Action 产生一个 Event，每个 Observation 也是一个 Event。Controller 消费 Event 流，驱动状态转移。这个设计让所有操作可重放（replay）、可审计。你有一个完整的 "event log"，可以从头重放整个 Agent 执行过程。

2. **状态显式化**：`RUNNING` / `AWAITING_USER` / `PAUSED` / `ERROR` 等状态区分清晰。用户交互（等待输入）和错误恢复是不同的状态，便于 UI 展示不同的交互模式。

3. **Agent-Controller 分离**：Agent 只产生 Action，Controller 负责执行 Action 并产生 Observation。Agent 不直接操作文件系统或执行命令，隔离性做得好。这让 Agent 可以被 Sandbox 化部署。

4. **多 Agent 架构预留**：事件流的设计让多个 Agent 可以并行消费事件，为 Agent Swarm 留下了扩展空间。

**OpenHands 状态机的问题**：

1. **状态爆炸（State Explosion）**：每个细粒度操作（读文件、写文件、执行命令）都产生 Event。一小时 coding session 轻松产生上千个 Event。状态机虽然是显式的，但 Event 流里隐含了大量"微状态"，真正的 Agent 心智模型不是这几个枚举能概括的。

2. **Controller 成为瓶颈**：Controller 承担了状态机调度 + Action 执行 + Observation 路由 + 安全沙箱，职责过重。一个 Controller 挂了，所有同一 sandbox 的 Agent 全挂。

3. **缺乏任务级状态**：状态机只描述了"Agent 在运行、在等待用户、出错了"，但没有描述"正在做步骤 3/7"。任务语义（"我在重构 auth 模块" vs "我在修一个 typo"）没有进入状态机，只存在 Event 流里，需要去解析 Event 才能知道。

4. **恢复机制不优雅**：`ERROR` 状态下，恢复策略是重试最后一个 Action。但如果错误原因是环境变化（端口被占用、文件被外部修改），重试会继续失败。恢复需要上下文，但状态机不携带足够上下文。

5. **等待用户输入的阻塞模型**：当状态进入 `AWAITING_USER`，整个 Controller 阻塞。用户可能在两个 session 里工作，但 OpenHands 的模型是一个 sandbox 对应一个 Agent。跨 session 协作需要额外设计。

6. **测试困难**：Event Sourcing 虽然能重放，但 Event 流和 Sandbox 状态绑定（文件系统、进程）。重放的前提是 Sandbox 环境完全一致，这在测试中很难保证。

**更优雅的实现方式：Actor Model + Task Graph**

我认为更好的设计是把 Agent 执行抽象为 Actor，每个 Actor 有独立的状态和 mailbox，Actor 之间通过消息通信。状态机不集中管理，而是分布在每个 Actor 内部。

```
┌──────────────────────────────────────────────────────┐
│                     Actor Model                       │
│                                                      │
│  ┌─────────────┐   Task消息   ┌──────────────────┐   │
│  │ Scheduler   │ ───────────→ │ TaskExecutor     │   │
│  │ Actor       │             │ Actor            │   │
│  │ (定时触发)   │             │ (执行单个任务)    │   │
│  └─────────────┘             └────────┬─────────┘   │
│                                       │              │
│                         分配子任务      │              │
│                                       ▼              │
│                              ┌──────────────────┐    │
│                              │ Worker Pool      │    │
│                              │ (多个Worker Actor)│    │
│                              │  ┌────┐ ┌────┐   │    │
│                              │  │ W1 │ │ W2 │   │    │
│                              │  └──┬─┘ └──┬─┘   │    │
│                              └─────┼──────┼──────┘    │
│                                    │      │           │
│                             工具调用│      │工具调用    │
│                                    ▼      ▼           │
│                              ┌──────────────────┐    │
│                              │ Tool Registry    │    │
│                              │ Actor            │    │
│                              │ (管理工具生命周期) │    │
│                              └──────────────────┘    │
└──────────────────────────────────────────────────────┘
```

**Actor Model 的优势**：

1. **天然隔离**：每个 Actor 有独立状态，一个 Worker 挂了不影响其他。监督树（Supervisor Tree）自动重启失败的 Actor。

2. **异步消息**：Actor 之间靠消息通信，不需要全局状态机。用户发新消息 → Scheduler Actor 收到消息 → 路由给对应 TaskExecutor。异步工具完成 → Tool Registry Actor 发消息给 Worker → Worker 继续。

3. **背压处理**：Actor mailbox 天然支持排队。busy 时消息在 mailbox 里等待，不会被丢弃，也不需要复杂的"busy 状态检查"。

4. **Task Graph 替代线性状态机**：不是 RUNNING → AWAITING → DONE 的线性流程，而是 DAG 任务图：

```
         ┌──────────┐
         │ 理解需求  │
         └────┬─────┘
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
 ┌──────┐ ┌──────┐ ┌──────┐
 │查代码 │ │查文档 │ │搜索   │   ← 这三步可以并行
 └──┬───┘ └──┬───┘ └──┬───┘
    │        │        │
    └────────┼────────┘
             ▼
        ┌──────────┐
        │ 修改代码  │
        └────┬─────┘
             ▼
        ┌──────────┐
        │  运行测试  │
        └────┬─────┘
        ┌────┼─────┐
        ▼    ▼     ▼
     通过  失败   需改
        │    │     │
        │    └──→ 修改代码 (回环)
        ▼
       完成
```

任务图里每个节点有自己的状态（pending/in_progress/done/failed），而不是一个全局状态机试图描述整个系统。

5. **任务语义内建**：每个 Task 节点携带：目标描述、输入依赖、输出 schema、超时、重试策略。状态不再是孤立的枚举值，而是携带完整上下文的对象。

**对比总结**：

| 维度 | OpenHands 状态机 | Actor + Task Graph |
|------|-----------------|-------------------|
| 状态模型 | 全局单一状态机 | 分布式 Actor 状态 + DAG 任务图 |
| 并行性 | 单 Controller 串行 | 多 Worker Actor 并行 |
| 错误隔离 | Controller 级 | Actor 级（监督树） |
| 任务语义 | 需从 Event 流解析 | 内建于 Task 节点 |
| 忙时处理 | 阻塞等待 | Mailbox 排队 |
| 可测试性 | 依赖 Sandbox 精确重放 | Actor 级单元测试 + 集成测试 |
| 复杂度 | 中（状态枚举可控） | 高（分布式系统问题） |

**给实战项目的建议**：如果你的 Agent 只有几个工具，5-6 步 loop，用简单的线性状态机（如本项目 `idle` → `busy` → `idle`）就够了。当工具数超过 50、任务涉及多步并行、需要跨 session 持久化任务时，再考虑 Actor Model。不要为了架构优雅而过早引入分布式复杂度。
