import os
import json
import subprocess
from openai import OpenAI

API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL_NAME = "qwen3.5-plus"

client = OpenAI(
    api_key=API_KEY,
    base_url=MODEL_URL
)

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
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

def execute_bash(command):
    """该函数执行一个bash命令，并返回标准输出和标准错误 """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def read_file(path):
    """该函数读取指定路径的文件内容并返回"""
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    """该函数将内容写入指定路径的文件，并返回确认消息"""
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote to {path}"


# dict，将工具名称映射到对应的函数,后续可以通过字符串名称动态调用函数
functions = {"execute_bash": execute_bash, "read_file": read_file, "write_file": write_file}


def run_agent(user_message, max_iterations=5):
    """该函数运行AI agent，处理用户消息，并根据需要调用工具
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


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    # 运行代理并打印结果
    print(run_agent(task))
