import os
import json
import subprocess
import sys
from datetime import datetime
from typing import Any
from openai import OpenAI

# 初始化Qwen客户端（通过DashScope API）
API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL_NAME = "qwen3.5-plus"

client = OpenAI(
    api_key=API_KEY,
    base_url=MODEL_URL
)

# 新增一个记忆文件
MEMORY_FILE = "agent_memory.md"

# 定义可用的工具列表。每个工具是一个dict，描述工具的类型type、工具函数fuction(含名称、描述和参数)。这里均是OpenAI Function Calling的标准格式规范。
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
            "description": "Execute a bash command on the system",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    }
]

def execute_bash(command):
    """该函数执行一个bash命令，并返回标准输出和标准错误 """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {str(e)}"

def read_file(path):
    """该函数读取指定路径的文件内容并返回"""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"

def write_file(path, content):
    """该函数将内容写入指定路径的文件，并返回确认消息"""
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {str(e)}"

# dict，将工具名称映射到对应的函数,后续可以通过字符串名称动态调用函数
available_functions = {
    "execute_bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file
}

def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """该函数解析工具调用的参数字符串"""
    if not raw_arguments:
        return {}
    try:
        parsed = json.loads(raw_arguments)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as error:
        return {"_argument_error": f"Invalid JSON arguments: {error}"}


# 简单的记忆管理系统，用于记忆读取和保存
def load_memory():
    """该函数加载记忆文件的内容"""
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
    """该函数保存任务和结果到记忆文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n## {timestamp}\n**Task:** {task}\n**Result:** {result}\n"           # 记录包括：时间戳、任务内容、结果内容
    try:
        with open(MEMORY_FILE, 'a') as f:
            f.write(entry)
    except:
        pass

def create_plan(task):
    """该函数根据任务创建执行计划"""
    print("[Planning] Breaking down task...")
    response = client.chat.completions.create(
        model=CHAT_MODEL_NAME, 
        messages=[
            # 系统提示，要求将任务分解成3-5个简单、可执行的步骤，并以JSON数组字符串的格式返回
            {"role": "system", "content": "Break down the task into 3-5 simple, actionable steps. Return as JSON array of strings."},
            {"role": "user", "content": f"Task: {task}"}
        ],
        response_format={"type": "json_object"}                 # 强制返回格式为JSON对象，LLM会按照这个格式生成内容，方便后续解析
    )
    try:
        plan_data = json.loads(response.choices[0].message.content)
        if isinstance(plan_data, dict):
            steps = plan_data.get("steps", [task])
        elif isinstance(plan_data, list):
            steps = plan_data
        else:
            steps = [task]
        print(f"[Plan] {len(steps)} steps created")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        return steps
    except:
        return [task]

def run_agent_step(task, messages, max_iterations=5):
    """该函数运行agent的单个步骤，处理任务并根据需要调用工具"""
    messages.append({"role": "user", "content": task})
    actions = []
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message
        messages.append(message)
        if not message.tool_calls:
            return message.content, actions, messages
        for tool_call in message.tool_calls:
            function_payload = getattr(tool_call, "function", None)
            if function_payload is None:
                continue
            function_name = str(getattr(function_payload, "name", ""))          # 工具调用名称
            raw_arguments = str(getattr(function_payload, "arguments", ""))     # 工具调用参数，JSON字符串
            function_args = parse_tool_arguments(raw_arguments)                 # 解析后的工具调用参数，字典格式
            print(f"[Tool] {function_name}({function_args})")
            function_impl = available_functions.get(function_name)
            if function_impl is None:
                function_response = f"Error: Unknown tool '{function_name}'"
            elif "_argument_error" in function_args:
                function_response = f"Error: {function_args['_argument_error']}"
            else:
                function_response = function_impl(**function_args)          # 调用工具函数并获取结果
                actions.append({"tool": function_name, "args": function_args})
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": function_response})
    return "Max iterations reached", actions, messages

def run_agent_plus(task, use_plan=False):
    """该函数运行增强版agent，支持计划分解和记忆功能"""

    # 加载记忆并构建系统提示，包含之前的上下文信息
    memory = load_memory()
    system_prompt = "You are a helpful assistant that can interact with the system. Be concise."
    if memory:
        system_prompt += f"\n\nPrevious context:\n{memory}"

    messages = [{"role": "system", "content": system_prompt}]
    if use_plan:
        steps = create_plan(task)           # 执行计划分解，获取步骤列表
    else:
        steps = [task]
    all_results = []                # 保存每个步骤的结果，最终合并成完整的结果返回给用户，包括最终结果和每个步骤的执行细节

    # 依次执行每个步骤，并收集结果
    for i, step in enumerate(steps, 1):
        if len(steps) > 1:
            print(f"\n[Step {i}/{len(steps)}] {step}")
        result, actions, messages = run_agent_step(step, messages)          
        all_results.append(result)          
        print(f"\n{result}")
    final_result = "\n".join(all_results)           
    save_memory(task, final_result)         # 将任务和结果保存到记忆中，供后续对话参考
    return final_result

if __name__ == "__main__":
    use_plan = "--plan" in sys.argv
    if use_plan:
        sys.argv.remove("--plan")
    if len(sys.argv) < 2:
        print("Usage: python agent-plus.py [--plan] 'your task here'")
        print("  --plan: Enable task planning and decomposition")
        sys.exit(1)
    task = " ".join(sys.argv[1:])
    run_agent_plus(task, use_plan=use_plan)
