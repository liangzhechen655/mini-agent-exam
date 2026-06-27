# Agent Notes

## Runtime

最小 Agent Runtime 的核心不是某个框架，而是一个稳定循环：接收用户输入，构造上下文，交给 LLM 决策，解析 LLM 输出，执行工具，把工具结果写回上下文，再继续循环直到得到最终答案。

## Context

上下文里应该放对当前回复有帮助的信息：最近几轮用户输入、最终回复、重要工具结果、会话摘要和必要 session state。Agent 的详细 thought 更适合进 trace，不适合反复塞回 LLM 上下文。

## Memory

短期状态放 session state，例如待办、最近查询城市。长期 memory 需要经过抽取、去重、权限控制和召回排序，不能把所有聊天原文无脑塞给模型。
