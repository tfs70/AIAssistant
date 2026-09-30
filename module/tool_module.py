from typing import Any
from mcp import ClientSession
from module.mcp_module import (
    call_tool,
    get_tools,
)


async def get_available_tools(
    session: ClientSession,
) -> list[Any]:
    """دریافت Toolهای موجود در MCP"""

    tools = await get_tools(
        session=session,
    )

    return tools


async def execute_tool(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """اجرای یک Tool"""

    result = await call_tool(
        session=session,
        tool_name=tool_name,
        arguments=arguments,
    )

    return result