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
