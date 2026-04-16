## 目录结构

```text
.
├─ agent.py                  # 原版+注释
├─ agent-plus.py             # 原版+注释
├─ agent-claudecode.py       # 原版+注释
├─ agent-paperread.py        # 实践应用
├─ MCPClient.py              # MCP 客户端
├─ .agent/
│  ├─ mcp.json               # MCP server 配置
│  ├─ rules/                 # 规则文件
│  └─ skills/                # 技能文件
├─ papers/                   # arXiv 下载与本地论文目录
└─ requirements.txt
```

## 环境
- Python 3.10+，建议使用虚拟环境
- 联网（调用 LLM API 与 OCR / arXiv 工具）

## 安装

```bash
pip install -r requirements.txt
```

## 环境变量
至少需要配置以下变量：

- DASHSCOPE_API_KEY：Qwen/DashScope API Key
- PADDLE_OCR_TOKEN：PaddleOCR 接口 Token（用于 PDF OCR）

## 快速开始
本地 PDF 阅读与总结：
```bash
python agent-paperread.py "帮我读papers目录下的论文并总结贡献、问题背景、方法、实验与局限性"
```

arXiv 检索 + 阅读：
```bash
python agent-paperread.py "帮我找这个月关于智能体记忆系统的预印论文，并挑一篇最相关的阅读"
```

复杂任务拆解执行：

```bash
python agent-paperread.py --plan "先检索多智能体协作论文，再对比三篇代表作"
```