import base64
import os
import json
import subprocess
import sys
import glob as glob_module
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from openai import OpenAI
from MCPClient import MCPClient 

API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL_NAME = "qwen3.5-plus"
PADDLE_OCR_API_URL = "https://b8w6xdj11as7r23e.aistudio-app.com/layout-parsing"
PADDLE_OCR_TOKEN = os.getenv("PADDLE_OCR_TOKEN")

client = OpenAI(
    api_key=API_KEY,
    base_url=MODEL_URL
)

MEMORY_FILE = "paper_reader_memory.md"     # 记忆文件
RULES_DIR = ".agent/rules"          # 规则文件目录
SKILLS_DIR = ".agent/skills"        # SKILLS文件目录
MCP_CONFIG = ".agent/mcp.json"      # MCP工具配置文件路径

current_plan = []
plan_mode = False
mcp_client = MCPClient()

# 基础工具定义，使用OpenAI/Qwen函数调用格式。
# 这些工具由LLM根据任务需要选择调用，参数使用JSON schema描述。
base_tools = [
    {"type": "function", "function": {"name": "read", "description": "Read file with line numbers", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write content to file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "glob", "description": "Find files by pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search files for pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "ocr_pdf", "description": "Use the PaddleOCR layout-parsing API to OCR a PDF and return the full recognized markdown content. Best for scanned papers, image-based PDFs, and complex layouts. The tool only accepts .pdf files and returns all pages in order.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Path to the PDF file to OCR"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "plan", "description": "Break down complex task into steps and execute sequentially", "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}}}
]

def read(path, offset=None, limit=None):
    """读取文件内容，并按行号返回部分结果"""
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        start = offset if offset else 0
        end = start + limit if limit else len(lines)
        numbered = [f"{i+1:4d} {line}" for i, line in enumerate(lines[start:end], start)]
        return ''.join(numbered)
    except Exception as e:
        return f"Error: {str(e)}"

def write(path, content):
    """将文本内容写入文件"""
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {str(e)}"


def glob(pattern):
    """按pattern(通配符)查找文件，并按修改时间降序返回"""
    try:
        files = glob_module.glob(pattern, recursive=True)   # 使用glob模块查找文件，支持递归
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return '\n'.join(files) if files else "No files found"
    except Exception as e:
        return f"Error: {str(e)}"

def grep(pattern, path="."):
    """在指定路径中递归搜索文本模式"""
    try:
        result = subprocess.run(f"grep -r '{pattern}' {path}", shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout if result.stdout else "No matches found"
    except Exception as e:
        return f"Error: {str(e)}"

def bash(command):
    """执行shell命令并返回输出结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {str(e)}"


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

def plan(task):
    """该函数生成一个任务执行计划，并将步骤保存到当前计划中，同plus版本"""
    global current_plan, plan_mode
    if plan_mode:
        return "Error: Cannot plan within a plan"
    print(f"[Plan] Breaking down: {task}")
    response = client.chat.completions.create(
        model=CHAT_MODEL_NAME,
        messages=[
            {"role": "system", "content": "Break task into 3-5 steps. Return JSON with 'steps' array."},
            {"role": "user", "content": task}
        ],
        response_format={"type": "json_object"}
    )
    try:
        plan_data = json.loads(response.choices[0].message.content)
        steps = plan_data.get("steps", [task])
        current_plan = steps
        print(f"[Plan] Created {len(steps)} steps")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        return f"Plan created with {len(steps)} steps. Executing now..."
    except:
        return "Error: Failed to create plan"


available_functions = {
    "read": read,
    "write": write,
    "glob": glob,
    "grep": grep,
    "bash": bash,
    "ocr_pdf": ocr_pdf,
    "plan": plan,
}

def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    if not raw_arguments:
        return {}
    try:
        parsed = json.loads(raw_arguments)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as error:
        return {"_argument_error": f"Invalid JSON arguments: {error}"}

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return ""
    try:
        with open(MEMORY_FILE, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            return '\n'.join(lines[-50:]) if len(lines) > 50 else content
    except:
        return ""

def save_memory(task, result):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n## {timestamp}\n**Task:** {task}\n**Result:** {result}\n"
    try:
        with open(MEMORY_FILE, 'a') as f:
            f.write(entry)
    except:
        pass

def load_rules():
    """该函数加载规则文件的内容，并返回合并后的字符串"""
    rules = []
    if not os.path.exists(RULES_DIR):
        return ""
    try:
        for rule_file in Path(RULES_DIR).glob("*.md"):
            with open(rule_file, 'r') as f:
                rules.append(f"# {rule_file.stem}\n{f.read()}")
        return "\n\n".join(rules) if rules else ""
    except:
        return ""

def load_skills():
    """该函数加载SKILL文件的内容，并返回技能列表"""
    skills = []
    if not os.path.exists(SKILLS_DIR):
        return []
    try:
        for skill_file in Path(SKILLS_DIR).glob("*.json"):
            with open(skill_file, 'r') as f:
                skills.append(json.load(f))
        return skills
    except:
        return []

def load_mcp_tools():
    """该函数加载MCP工具配置，并返回工具列表"""
    mcp_tools = []
    try:
        all_tools = mcp_client.list_all_tools_sync()
    except Exception as e:
        print(f"[MCP] load failed: {e}")
        return mcp_tools

    for _, tools in all_tools.items():
        for tool in tools:
            mcp_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                    },
                }
            )
    return mcp_tools


def print_tools_summary(tools):
    print(f"\n[Tools] Total: {len(tools)}")
    for i, tool_item in enumerate(tools, 1):
        fn = tool_item.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}) or {}
        props = list((params.get("properties") or {}).keys())
        required = params.get("required") or []

        print(f"{i:02d}. {name}")
        if desc:
            print(f"    desc: {desc}")
        if props:
            print(f"    args: {', '.join(props)}")
        else:
            print("    args: (none)")
        if required:
            print(f"    required: {', '.join(required)}")



def run_agent_step(messages, tools, max_iterations=10):
    """该函数运行agent，处理工具调用并返回最终结果"""
    global current_plan, plan_mode
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message
        print(f"[Debug] Message in this iteration:\n{message}")

        messages.append(message)
        if not message.tool_calls:
            return message.content, messages
        for tool_call in message.tool_calls:
            function_payload = getattr(tool_call, "function", None)
            if function_payload is None:
                continue
            function_name = str(getattr(function_payload, "name", ""))
            raw_arguments = str(getattr(function_payload, "arguments", ""))
            function_args = parse_tool_arguments(raw_arguments)
            print(f"[Tool] {function_name}({function_args})")
            function_impl = available_functions.get(function_name)
            if "_argument_error" in function_args:
                function_response = f"Error: {function_args['_argument_error']}"
            elif function_name == "plan" and function_impl is not None:
                plan_mode = True
                function_response = function_impl(**function_args)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": function_response})
                if current_plan:
                    results = []
                    for i, step in enumerate(current_plan, 1):
                        print(f"\n[Step {i}/{len(current_plan)}] {step}")
                        messages.append({"role": "user", "content": step})
                        result, messages = run_agent_step(messages, [t for t in tools if t["function"]["name"] != "plan"])
                        results.append(result)
                        print(f"\n{result}")
                    plan_mode = False
                    current_plan = []
                    return "\n".join(results), messages
            elif function_impl is not None:
                function_response = function_impl(**function_args)
            else:
                # Not a built-in tool: try MCP tools by function name.
                function_response = mcp_client.call_tool_sync(function_name, function_args)
                function_response = json.dumps(function_response, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": function_response})
            print(f"[Debug] Tool response:\n{function_name}: {function_response}")
    return "Max iterations reached", messages

def run_agent_claudecode(task, use_plan=False):
    """该函数运行增强版ClaudeCode Agent，支持记忆、规则、技能、MCP和计划功能"""
    global plan_mode, current_plan
    print("[Init] Loading ClaudeCode features...")
    memory = load_memory()
    rules = load_rules()
    skills = load_skills()
    mcp_tools = load_mcp_tools()
    all_tools = base_tools + mcp_tools              # 这里新增mcp提供的工具列表，
    context_parts = [
        "You are a paper reading assistant. Help users read, summarize, compare, and critique academic papers. Be concise and evidence-based."
    ]

    print_tools_summary(all_tools)

    # 根据加载的内容构建系统提示，包含规则、技能和记忆等信息，提供给LLM使用
    if rules:
        context_parts.append(f"\n# Rules\n{rules}")
        print(f"[Rules] Loaded {len(rules.split('# '))-1} rule files")
    if skills:
        context_parts.append(f"\n# Skills\n" + "\n".join([f"- {s['name']}: {s.get('description', '')}" for s in skills]))
        print(f"[Skills] Loaded {len(skills)} skills")
    if mcp_tools:
        print(f"[MCP] Loaded {len(mcp_tools)} MCP tools")
    if memory:
        context_parts.append(f"\n# Previous Context\n{memory}")
    messages = [{"role": "system", "content": "\n".join(context_parts)}]
    if use_plan:
        plan_mode = True
        print(plan(task))
        results = []
        for i, step in enumerate(current_plan, 1):
            print(f"\n[Step {i}/{len(current_plan)}] {step}")
            messages.append({"role": "user", "content": step})
            result, messages = run_agent_step(messages, [t for t in all_tools if t["function"]["name"] != "plan"])
            results.append(result)
            print(f"\n{result}")
        plan_mode = False
        current_plan = []
        final_result = "\n".join(results)
    else:
        messages.append({"role": "user", "content": task})
        final_result, messages = run_agent_step(messages, all_tools)
        print(f"\n{final_result}")
    save_memory(task, final_result)
    return final_result

if __name__ == "__main__":
    use_plan = "--plan" in sys.argv
    if use_plan:
        sys.argv.remove("--plan")
    if len(sys.argv) < 2:
        print("Usage: python agent-math.py [--plan] 'your task'")
        print("  --plan: Enable task planning")
        print("\nFeatures: Paper Memory, Rules, Skills, MCP, Plan tool")
        sys.exit(1)
    task = " ".join(sys.argv[1:])
    run_agent_claudecode(task, use_plan=use_plan)




