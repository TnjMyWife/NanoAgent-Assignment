# 论文阅读助手配置说明

本目录用于演示在 `agent-math.py` 中如何体现四类扩展机制：记忆文件、规则文件、Skills、MCP。

## 1. 记忆文件
- 文件：`paper_reader_memory.md`
- 来源：`agent-math.py` 中 `MEMORY_FILE` 配置
- 作用：记录历史任务与输出，用于下一轮对话补充上下文

## 2. 规则文件
- 目录：`.agent/rules/`
- 示例：`paper_reading.md`
- 作用：约束输出格式和分析质量（证据优先、固定对比维度等）

## 3. Skills 文件
- 目录：`.agent/skills/`
- 示例：`paper_workflow.json`
- 作用：给模型提供领域化工作流模板，提升稳定性

## 4. MCP 配置
- 文件：`.agent/mcp.json`
- 当前状态：`disabled=true`（占位模板）
- 作用：接入外部能力（论文检索、文献库写入等）

## 快速试运行
```bash
python agent-math.py --plan "请阅读并总结 Transformer 论文，按贡献、方法、实验、局限性输出"
```

```bash
python agent-math.py "对比 ResNet 与 ViT 的核心思想和适用场景"
```

如果要启用 MCP，把 `.agent/mcp.json` 里对应 server 的 `disabled` 改成 `false`，并提供真实的 MCP server 可执行文件。
