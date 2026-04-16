# NanoAgent 分析报告

NanoAgent 是一个基于LLM的智能代理框架，支持工具调用、记忆系统、任务规划等功能。目前的项目包含三个递进版本的agent实现，从简单到复杂。

1. **agent.py**：基础最简单的实现，支持基本工具调用
2. **agent-plus.py**：在基础版本上增加记忆系统和任务分解
3. **agent-claudecode.py**：最复杂的实现，支持规则、技能、MCP工具等高级功能

详细注释和实践代码已放在仓库：
## 1. 源码注释：

### agent.py 分析

最基础的AI代理实现，LLM+支持三个基本工具：执行bash命令、读取文件、写入文件。

#### 1. 工具定义
```python
# 定义可用的工具列表。每个工具是一个dict，描述工具的类型type、工具函数fuction(含名称、描述和参数)。这里均是OpenAI Function Calling的标准格式规范，也是LLM的返回的工具调用格式
# LLM读到这些信息后，就能在对话中判断是否需要调用某个工具；提取出正确的参数值；按照规定的格式调用函数
# 这里定义了三个工具：
#   execute_bash：执行shell命令并返回输出。
#   read_file：读取指定文件的全部内容。
#   write_file：将内容写入指定文件。
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    # ... 其他工具
]
```

#### 3. 工具函数实现
- `execute_bash(command)`: 执行shell命令，返回输出
- `read_file(path)`: 读取文件内容
- `write_file(path, content)`: 写入文件内容

#### 4. 主agent执行逻辑
```python
# dict，将工具名称映射到对应的函数,后续可以通过字符串名称动态调用函数
functions = {"execute_bash": execute_bash, "read_file": read_file, "write_file": write_file}


def run_agent(user_message, max_iterations=5):
    """该函数运行AI代理，处理用户消息，并根据需要调用工具
    参数：
        user_message: 用户输入的消息
        max_iterations: 最大迭代次数，默认值为5
    返回：最终的响应内容或错误消息
    """

    # 初始化消息列表，包含系统提示和用户消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": user_message},
    ]
    # 循环迭代处理消息，直到达到最大迭代次数
    for _ in range(max_iterations):
        # 调用API生成响应
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=messages,
            tools=tools,
        )
        # 获取响应中的消息，choices是列表，API同时生成多个候选回答，[0]取概率最高
        message = response.choices[0].message
        # 将消息添加到消息列表中
        messages.append(message)
        
        if not message.tool_calls:
            # 如果没有工具调用，返回内容作为最终响应。说明LLM最后一次的生成能够根据历史结果直接给出答案了，不需要再调用工具了
            return message.content
        # 处理每个工具调用
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"[Tool] {name}({args})")
            # 检查工具是否存在
            if name not in functions:
                result = f"Error: Unknown tool '{name}'"
            else:
                # 调用对应的函数
                result = functions[name](**args)
            # 将工具结果添加到消息列表中
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
    # 如果达到最大迭代次数，返回错误消息
    return "Max iterations reached"
```

总体流程如下：

1. 初始化对话消息
2. 调用Qwen API生成响应
3. 如果有工具调用，执行相应函数
4. 将工具结果反馈给模型
5. 重复直到获得最终答案或达到迭代上限

### agent-plus.py 分析

在基础代理基础上增加记忆系统和任务规划功能。支持保存执行历史，并能将复杂任务分解为多个步骤执行。

#### 1. 简易的记忆系统
```python
MEMORY_FILE = "agent_memory.md"
def load_memory():
    # 加载最近50行记忆内容
def save_memory(task, result):
    # 保存任务执行结果到记忆文件
```

#### 2. 任务规划
```python
def create_plan(task):
    # 该函数根据任务创建执行计划，使用LLM将复杂任务分解为3-5个步骤

def run_agent_plus(task, use_plan=False):
    # 主agent执行逻辑，支持任务规划的模式，先将复杂任务分解，再依次执行分解的任务，得到的结果存入记忆文件中
```

#### 3. 增强的agent循环
```python
def run_agent_step(task, messages, max_iterations=5):
    # 该函数LLM执行单个子任务，直到生成结果或超过最大迭代次数
```

核心改进在于：**记忆持久化**，保存任务历史供后续使用；**任务分解**，复杂任务自动拆分为可管理步骤；**状态管理**，跟踪执行进度和结果

### agent-claudecode.py 分析

较完整的代理实现，类似ClaudeCode，支持多种高级功能：记忆、规则、SKILLS、MCP工具、计划等。

#### 1. 规则系统
```python
RULES_DIR = ".agent/rules"
def load_rules():
    # 从.md文件加载自定义规则
```

#### 2. SKILLS
```python
SKILLS_DIR = ".agent/skills"
def load_skills():
    # 从.json文件加载技能配置
```

#### 3. MCP工具集成

MCP是模型上下文协议。通过外部独立的MCP服务器提供工具。允许Agent从配置文件动态加载工具定义，有着标准化的接口，遵循统一的工具描述格式。通过MCP，NanoAgent支持任意第三方工具，而无需修改代码。

```python
MCP_CONFIG = ".agent/mcp.json"
def load_mcp_tools():
    # 加载MCP工具
```

#### 4. 增强的基础工具集
```python
base_tools = [
    {"name": "read", "description": "Read file with line numbers"},
    {"name": "write", "description": "Write content to file"},
    {"name": "edit", "description": "Replace string in file"},
    {"name": "glob", "description": "Find files by pattern"},
    {"name": "grep", "description": "Search files for pattern"},
    {"name": "bash", "description": "Run shell command"},
    {"name": "plan", "description": "Break down complex task into steps"},
]
def read(path, offset=None, limit=None):
    """读取文件内容，并按行号返回部分结果"""
    
def write(path, content):
    """将文本内容写入文件"""
    
def glob(pattern):
    """按pattern(通配符)查找文件，并按修改时间降序返回"""
    
def grep(pattern, path="."):
    """在指定路径中递归搜索文本模式"""

def bash(command):
    """执行shell命令并返回输出结果"""

def plan(task):
    """该函数生成一个任务执行计划，并将步骤保存到当前计划中，同plus版本"""
```

#### 5. 上下文构建
```python
def run_agent_claudecode(task, use_plan=False):
    # 该函数运行增强版ClaudeCode Agent，支持记忆、规则、技能、MCP和计划功能
    memory = load_memory()
    rules = load_rules()
    skills = load_skills()
    mcp_tools = load_mcp_tools()
    	
    # 根据加载的内容构建系统提示，包含规则、技能和记忆等信息，提供给LLM使用
    context_parts = ["You are a helpful assistant..."]
    if rules: 
        context_parts.append(f"\n# Rules\n{rules}")
    if skills: 
        context_parts.append(f"\n# Skills\n{...}")
    if memory: 
        context_parts.append(f"\n# Previous Context\n{memory}")
    
    ...
```



## 2.原理分析：

### 核心原理

NanoAgent本质是遵循LLM 驱动的工具调用智能体，是一个基于ReAct（Reason+Act）推理执行框架的简易实现实现。NanoAgent利用LLM作为大脑，负责理解任务、推理决策、生成Function Calling工具调用参数、总结结果。利用工具作为手脚，封装文件操作、Shell 执行、文件搜索等系统能力，供大模型按需调用。在plus和claudecode版本中，增加了记忆 + 规则 + 技能的实现，通过本地文件实现长期记忆、自定义规则、扩展技能，提升智能体上下文一致性和专业性。同时支持任务规划，将复杂任务自动拆解为子步骤，分步执行，解决LLM在对长复杂任务能力不足。

**核心工作流**：接收任务 → 加载记忆 / 规则 / 技能 → LLM 推理 → 调用工具 → 执行结果回传 LLM → 循环决策 → 完成任务 → 保存记忆。



### 核心架构

NanoAgent的核心架构分为4部分：

交互入口层：NanoAgent采用最简易的CLI与用户进行交互，通过命令行启动、参数解析来获取提问并

智能体核心调度层：主要在run_agent_claudecode，总控流程，负责统一管理上下文、工具集合、循环迭代、计划执行，是智能体的中枢神经

工具执行模块：封装多个基础工具，供LLM使用Function Calling机制实现与系统交互。工具的使用是Agent解决LLM只会说不会做的关键。

能力扩展层：负责记忆(保存历史上下文)/规则（LLM约束）/技能（拓展专业能力）/MCP工具（加载第三方工具配置）的加载。主要集成到系统提示词中供LLM参考，进一步增强Agent的功能。

LLM模块：利用OpenAI API，实现LLM交互对话。



### 关键算法

#### 1. Function Calling机制

这是智能体的**核心能力**，完全遵循 OpenAI 函数调用规范：

- 定义base_tools：用 JSON Schema 描述每个工具的名称、功能、参数，供 LLM 理解；
- 工具映射表available_functions：将工具名称与 Python 函数绑定，实现动态调用；
- 执行流程：LLM 返回tool_calls→ 解析参数 → 执行函数 → 结果回传 LLM。

#### 2.ReAct 多轮迭代

`run_agent_step`是核心执行函数，实现类似ReAct循环：

```python
for _ in range(max_iterations):
    1. 调用LLM获取响应
    2. 无工具调用 → 直接返回结果
    3. 有工具调用 → 批量执行所有工具
    4. 工具结果追加到对话上下文
    5. 循环迭代，直到任务完成/达到最大迭代次数
```

####  3.任务规划

解决复杂任务拆解问题。利用LLM 将任务拆分为 3-5 个子步骤，返回 JSON 格式步骤；执行逻辑： 分步执行子任务 → 汇总结果 → 清空计划；通过这样来降低LLM的单步任务复杂度，提升整个复杂任务成功率。



## 3.实践

### 应用案例：论文阅读助手

本次基于nanoAgent的基本框架实现了一个简单的论文阅读助手，主要面向两类简单场景：

1. 用户提供本地PDF，希望直接完成论文解析、摘要和结构化阅读。
2. 用户希望在arxiv上检索目标主题论文，并进一步下载、阅读与比较。

围绕这两个场景，agent实际上先判断来源，再选择工具链：本地PDF走OCR，arxiv论文走检索与下载工具，最后统一进入LLM的阅读和总结流程。这样把找论文和读论文拆成两个阶段，避免模型在没有原始内容的情况下直接生成结论。

#### MCP 接入

本项目的arxiv能力通过 MCP 工具接入。实际上目前MCP大部分使用需要由MCP客户端对服务器发送请求，而不是像当前nanoAgent中的硬编码进配置文件。mcp.json一般注册MCP相关服务器。智能体启动时，MCP Client会读取MCP服务器的配置并启动，同时向服务器建立连接获取可访问工具，这些工具和基础工具一起传给 LLM，让模型在同一个 function call机制下决定是否调用。本次使用到开源arxiv MCP工具([blazickjp/arxiv-mcp-server: A Model Context Protocol server for searching and analyzing arXiv papers](https://github.com/blazickjp/arxiv-mcp-server))，其提供了最新学术论文的搜索和分析。配置如下：

```json
{
  "mcpServers": {
    "arxiv-mcp-server": {
      "command": "uv",
      "args": [
        "tool",
        "run",
        "arxiv-mcp-server",
        "--storage-path",
        "./papers"
      ]
    }
  }
}
```

此外，需要实现一个建议的MCP Client以连接服务器并获取、调用工具。参考了MCP官网，实现了一个极简的MCP客户端。

```python
# 一个简单的MCP客户端实现，支持从配置文件连接多个MCP服务器，并调用工具。
# 基本思路：
# 1. 从指定路径加载mcp.json配置，获取MCP服务器列表和连接参数。
# 2. 对于每个未禁用的服务器配置，启动一个stdio客户端会话，并保存会话对象。
# 3. 提供接口列出所有连接的服务器上的工具信息，以及按工具名调用工具并返回结果。 

# 实现参考mcp官网：https://mcp-docs.cn/docs/develop/build-client

import argparse
import asyncio
import json
import os
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, config_path=".agent/mcp.json"):
        # MCP 配置文件路径，默认读取仓库内的 .agent/mcp.json。
        self.config_path = config_path
        # AsyncExitStack用来统一管理多个异步上下文资源：stdio连接和 session。
        # 这样在关闭客户端时，可以一次性释放所有已经打开的 MCP 连接。
        self.exit_stack = AsyncExitStack()
        # 记录已经连接好的 MCP 会话，key 是 server 名称，value 是对应的 ClientSession。
        self.sessions = {}          # server_name->ClientSession

    async def close(self):
        # 关闭所有通过 exit_stack 注册的异步资源，避免子进程和管道泄漏。
        await self.exit_stack.aclose()
        
    async def connect_all(self):
        """根据mcp.json配置连接所有MCP服务器，并建立会话。"""

        # 1) 读取 MCP 配置。
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        mcp_servers = config.get("mcpServers")

        # 2) 逐个遍历配置中的MCP server。
        # 每个server都可能对应一个独立的外部进程，因此这里按server维度创建连接。
        for server_name, server_cfg in mcp_servers.items():
            # 支持在配置里显式禁用某个server，方便保留模板或临时关停能力。
            if server_cfg.get("disabled", False):
                continue

            command = server_cfg.get("command")
            args = server_cfg.get("args", [])

            # 3) 构造该server运行时使用的环境变量。
            # 先复制当前进程环境，再按配置项覆盖，避免影响当前Python进程的全局环境。
            env_cfg = server_cfg.get("env", {})
            env = os.environ.copy()
            for k, v in (env_cfg or {}).items():
                # 配置文件里允许写变量占位符，因此这里做一次环境变量展开。
                env[k] = os.path.expandvars(str(v))

            # 4) 将command/args/env封装成MCP需要的stdio启动参数。
            # 这一步相当于告诉 MCP：如何启动目标 server 以及如何与它通信。
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env,
            )
            # 5) 使用stdio_client启动外部server，并拿到双向通信管道。
            # enter_async_context的作用是把这个连接注册到exit_stack，后续统一回收。
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            # 6) 基于stdio通道创建 MCP 会话。
            # ClientSession负责后续的 initialize、list_tools、call_tool等协议交互。
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            # 7) 初始化会话，完成与 server 的握手。
            # 只有 initialize 成功后，后续工具列表和工具调用才可用。
            await session.initialize()
            
            self.sessions[server_name] = session


    async def list_all_tools(self):
        """列出所有连接的MCP服务器上的工具信息。"""
        result = {}
        for server_name, session in self.sessions.items():
            resp = await session.list_tools()
            tools = []
            for tool in resp.tools:
                tools.append(
                    {
                        "name": getattr(tool, "name", ""),
                        "description": getattr(tool, "description", ""),
                        "input_schema": getattr(tool, "inputSchema", None),
                    }
                )
            result[server_name] = tools
        return result

    def _serialize_tool_result(self, raw_result):
        # MCP工具返回值在不同SDK版本下可能是不同对象形态。这里统一归一化为可JSON化的数据，方便上层直接写入对话上下文。
        if hasattr(raw_result, "model_dump"):
            return raw_result.model_dump()
        if hasattr(raw_result, "dict"):
            return raw_result.dict()
        if hasattr(raw_result, "content"):
            content = getattr(raw_result, "content")
            items = []
            for item in content or []:
                if hasattr(item, "model_dump"):
                    items.append(item.model_dump())
                elif hasattr(item, "dict"):
                    items.append(item.dict())
                else:
                    items.append(str(item))
            return {"content": items, "isError": getattr(raw_result, "isError", False)}
        
        return str(raw_result)

    async def call_tool(self, tool_name, args={}):
        """调用工具，并返回结果。"""

        # 查找提供该工具的服务器会话
        async def find_server_for_tool(tool_name):
            # 按server逐个查询工具列表，找到第一个包含目标工具的 server。
            for server_name, session in self.sessions.items():
                resp = await session.list_tools()
                tools = resp.tools
                if any(getattr(t, "name", "") == tool_name for t in tools):
                    return server_name, session
            return None, None

        # 1) 定位工具属于哪个 MCP server。
        server_name, session = await find_server_for_tool(tool_name)
        if not session:
            # 如果所有已连接 server 都没有该工具，直接报错给上层。
            raise ValueError(f"Tool '{tool_name}' not found in any connected MCP server.")

        # 2) 将参数发送给对应 MCP server，由server真正执行工具逻辑。
        resp = await session.call_tool(tool_name, args)
        return {
            "server": server_name,
            "tool": tool_name,
            "args": args,
            "result": self._serialize_tool_result(resp),
        }



    def list_all_tools_sync(self):
        """同步接口：列出所有连接的MCP服务器上的工具信息。"""
        async def runner():
            await self.connect_all()
            try:
                return await self.list_all_tools()
            finally:
                await self.close()

        return asyncio.run(runner())
    
    def call_tool_sync(self, tool_name, args=None):
        """同步接口：按工具名调用MCP工具，并返回结果。"""

        if args is None:
            args = {}

        async def runner():
            await self.connect_all()
            try:
                return await self.call_tool(tool_name, args)
            finally:
                await self.close()

        return asyncio.run(runner())
```

#### 额外的工具扩展

在现有基础工具之外，仍需补充了两个关键方向的能力。第一类是PDF阅读工具 `ocr_pdf`，它专门用于读取本地 PDF 文件转化为md。使用了PaddleOCR的接口，把PDF转成按页拼接的 Markdown 内容。目前它在PDF OCR的多个任务上表现最好。这个工具可以允许agent去处理扫描版PDF或排版复杂的论文，再让 LLM 进一步完成阅读和总结。

```python
def ocr_pdf(file_path):
    """调用 PaddleOCR layout-parsing 接口识别 PDF，并返回完整的 Markdown OCR 文本。

    这个工具只接受 PDF 文件，适合扫描版论文、图片型 PDF 或排版复杂的文档。
    返回值会按页顺序拼接全部识别结果，保留页码信息，方便后续总结、引用和分析。
    """

    if not file_path.lower().endswith(".pdf"):
        return "Error: ocr_pdf only supports .pdf files"
    if not PADDLE_OCR_TOKEN:
        return "Error: PADDLE_OCR_TOKEN is not set"

    try:
        # 读取整个PDF文件并做Base64编码。PaddleOCR的这个接口要求把文件内容直接作为JSON字段传给服务端。
        with open(file_path, "rb") as f:
            file_data = base64.b64encode(f.read()).decode("ascii")
        # 构造接口请求体。fileType=0 表示这里传入的是 PDF
        payload = {
            "file": file_data,
            "fileType": 0,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }

        # 构造请求头
        headers = {
            "Authorization": f"token {PADDLE_OCR_TOKEN}",
            "Content-Type": "application/json",
        }
        print(f"PaddleOCR识别PDF: {os.path.basename(file_path)}")

        # 生成标准 HTTP POST 请求对象，随后交给 urlopen 发送。
        request = urllib.request.Request(
            PADDLE_OCR_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            # 发起网络请求
            with urllib.request.urlopen(request, timeout=300) as response:
                response_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace") if error.fp else ""
            return f"Error: PaddleOCR request failed with HTTP {error.code}: {error_body or error.reason}"
        except urllib.error.URLError as error:
            return f"Error: PaddleOCR request failed: {error.reason}"

        try:
            # 解析OCR服务返回的JSON。根据接口定义，识别结果保存在result.layoutParsingResults字段，按页顺序排列的数组。
            response_data = json.loads(response_text)
            layout_results = response_data["result"]["layoutParsingResults"]
        except (KeyError, TypeError, json.JSONDecodeError):
            return "Error: PaddleOCR response format is invalid or incomplete"

        total_pages = len(layout_results)
        page_texts = []
        for index, page_result in enumerate(layout_results, 1):
            # 每一页的 markdown 文本保存在 markdown.text 中。
            markdown_text = ""
            if isinstance(page_result, dict):
                markdown_data = page_result.get("markdown", {})
                if isinstance(markdown_data, dict):
                    markdown_text = markdown_data.get("text", "") or ""

            page_texts.append(f"## Page {index}/{total_pages}\n{markdown_text}")
            print(f"PDF OCR 第 {index}/{total_pages} 页加载完成")
        # 拼接返回
        return "\n\n".join(page_texts)
    except Exception as e:
        return f"Error: {str(e)}"
```

另一个是 arxiv 阅读工具链。通过前面说的MCP获取。提供的工具包括：`search_papers` 用于搜索候选论文，`get_abstract` 用于先筛选相关性，`download_paper` 和 `read_paper` 用于获取完整正文，`semantic_search` 和 `citation_graph` 用于做语义扩展和引用扩展。这样模型面对“找最新论文”“找经典论文”“找某主题的综述替代项”“追踪一个研究方向”的请求时，能走不同的工具路径。

利用Function Call，检索、筛选、读取、总结这几个动作可以统一纳入一个闭环中，LLM 每轮都根据前一轮工具返回的结果决定下一步，而不是把所有步骤写死在Skills或规则中。对论文助手而言，这一点很重要，因为不同用户的问题往往处在不同阶段，有时候只是想找论文，有些本地有PDF，有些还想继续比较相关工作，所以必须允许不同的工具路径，而不能强行用一条固定流程覆盖所有场景。

#### 规则

为了让 LLM 不乱选工具，在规则里明确了任务路由：本地 PDF 必须先走 `ocr_pdf`，arXiv 论文必须先走 `search_papers` 或 `get_abstract`，不能把 `read`、`grep`、`bash` 当成论文理解工具来替代专用能力。规则还要求模型在输出时优先采用证据驱动的写法：先给论文元信息，再给三句话摘要，然后给方法、实验、局限性，最后再做跨论文对比或后续建议。规则的作用不只是约束表达，更重要的是约束行动。如果不先约束工具路径，LLM 很容易在 PDF 场景里直接凭上下文猜测内容，或者在arxiv场景里只看摘要就给出完整结论。规则把这些偏差提前压住，使整个系统更接近一个稳定agent，而不是一个只会生成文本的聊天模型。

```markdown
# 论文阅读规则

你是一个论文阅读助手。先判断用户要处理的对象，再选择正确工具链。

## 1. 任务路由
1. 如果用户给的是本地PDF(目前PDF都在papers目录下)，先用 `ocr_pdf` 做全文识别，再基于 OCR 结果进行阅读、总结、对比或批注。
2. 如果用户要找 arXiv 论文，先用 `search_papers` 搜索；必要时再用 `get_abstract` 过滤；确认相关后用 `download_paper` 或 `read_paper` 获取全文。
3. 如果用户已经给出 arXiv ID，优先用 `get_abstract` 判断是否相关，再决定是否 `download_paper`、`read_paper`、`citation_graph` 或 `semantic_search`。
4. 如果用户的需求同时包含“找论文 + 读论文”，先完成检索，再读取，再总结，不要跳过中间步骤。
5. 如果来源不明确，先追问或给出可执行的澄清方案，不要凭空假设论文来源。

## 2. PDF 处理规则
1. PDF只走 `ocr_pdf`，不要假设 `read`、`grep` 或 `bash` 可以替代 OCR。
2. `ocr_pdf` 返回的是按页拼接的 Markdown 内容，阅读时要保留页码信息，不要混淆不同页面的结论。
3. 如果 OCR 结果不完整、乱码或缺页，要明确说明质量问题，并建议重新识别或换更清晰的 PDF。
4. 对扫描版论文、图片型 PDF、复杂排版论文，优先信任 OCR 后的版面结构，而不是臆测原文。

## 3. arXiv 处理规则
1. 搜索arXiv时优先使用精确短语、字段限定和类别过滤，避免宽泛关键词堆叠。
2. 建议先用 `search_papers` 找候选，再用 `get_abstract` 过滤，再用 `download_paper` 或 `read_paper` 读取全文。
3. 如果用户要求“最新”“近期”“经典”，分别用 `sort_by: date`、`date_from`、`date_to` 进行约束。
4. 如果需要追踪主题，先用 `watch_topic` 创建监控，再用 `check_alerts` 拉取新结果。
5. 对于引用关系、相似论文、研究脉络，优先使用 `citation_graph`、`semantic_search`、`list_papers` 这类专用工具。

## 4. 阅读与分析规则
1. 先确认论文基本信息：标题、作者、年份、venue、arXiv ID 或 PDF 来源。
2. 所有结论都必须有可验证依据，优先引用标题、摘要、章节、段落或明确的原文句子。
3. 先给简短摘要，再给结构化分析，不要一上来输出长篇散文。
4. 如果证据不足，明确写出“不确定”，并指出缺失信息来自哪里。
5. 论文必须覆盖：问题定义、核心方法、实验设置、结果、局限性。
6. 对比多篇论文时，固定维度：任务、数据集、指标、复杂度、优缺点。
7. 不要编造实验数字、作者信息、引用、发布日期或结论。

## 5. 输出要求
1. 默认输出应紧凑、结构化、基于证据。
2. 如果用户没有特别要求，优先输出：摘要、问题背景、方法、实验、局限性、下一步建议。
3. 如果用户要求“解析论文”或“读论文”，优先输出可直接用于笔记的 Markdown 结构。
4. 如果用户要求对比论文，先统一比较维度，再给结论，不要只做逐篇复述。
```

#### 效果

下面给出两个简单的用例，用来观察 function call 的迭代循环以及智能体是否遵循规则。

##### 用例 1：本地 PDF 论文阅读

输入在本地目录下的一个PDF文件，“帮我读当前目录下论文并总结贡献、问题背景、方法和实验结果、局限性”。在这个场景里，智能体的理想执行链路是：

1. LLM 首先识别任务来源是本地 PDF，而不是 arxiv。

   <img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20260416215903236.png" alt="image-20260416215903236" style="zoom: 67%;" />



2. 模型发起 `ocr_pdf(file_path=...)` 调用，先把整篇 PDF 转成按页 Markdown。

<img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20260416215740456.png" alt="image-20260416215740456" style="zoom: 67%;" />

<img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20260416220215650.png" alt="image-20260416220215650" style="zoom: 67%;" />

3. OCR 结果返回后，LLM进入第二轮function call或直接进入总结轮，基于识别文本做总结和结构化分析。

   <img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20260416221407344.png" alt="image-20260416221407344" style="zoom: 67%;" />

##### 用例 2：arXiv主题检索与阅读

输入可以是“帮我找最近的智能体关于记忆系统的论文，并挑一篇最相关的阅读”：

1. LLM 先调用 `search_papers`，使用带字段和类别约束的 query 搜索候选论文。

   <img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20260416222344576.png" alt="image-20260416222344576" style="zoom: 80%;" />

   这里agent筛选时间从2025开始，因为本次的简单实现并没有给agent提供获取时间的工具，所以并不实时。

2. 选中目标后，再调用 `download_paper` 或 `read_paper` 获取全文。

   <img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20260416222835018.png" alt="image-20260416222835018" style="zoom:67%;" />

<img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20260416223222266.png" alt="image-20260416223222266" style="zoom: 67%;" />

从上面用例能很好地看出智能体是否遵循“先检索、后判断、再阅读”的原则。遵循原则时，模型会先确认相关性，再花代价下载全文，不会一上来就读一堆不相关论文；同时它会优先使用 arXiv 专用工具，而不是把任务退化成网页搜索或普通文件搜索。反过来，如果模型不稳定，常见问题就是 query 过宽、忽略类别过滤、直接下载大量论文，导致后续分析质量下降。从 function call 角度看，模型不是一次工具调用就结束，而是会在每轮根据工具反馈更新下一步动作。好的表现是每一步都有明确目的，例如先缩小候选集，再获取摘要，再决定是否下载。

#### 总结：

本次实践是基于 nanoAgent 的一个简单案例，核心目的是验证LLM结合 function call后的能力扩展：模型不再只是生成文本，而是可以通过工具完成网络检索、外部服务调用以及本地系统交互，并在ReAct模式形成“检索-筛选-读取-总结”的闭环；不足在于记忆系统仍较基础，尚未实现结构化、可检索的长期知识复用，一般来说可通过引入RAG对论文内容与分析结果进行向量化存储和语义检索，进一步提升连续任务中的准确性与稳定性。













