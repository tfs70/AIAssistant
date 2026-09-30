import os
import httpx

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://localhost:6111/mcp",
)

MCP_JWT_TOKEN = os.getenv(
    "MCP_JWT_TOKEN",
    "",
)


def create_http_client() -> httpx.AsyncClient:
    """ایجاد HTTP Client برای اتصال به MCP"""

    headers: dict[str, str] = {}

    if MCP_JWT_TOKEN:
        headers["Authorization"] = (
            f"Bearer {MCP_JWT_TOKEN}"
        )

    return httpx.AsyncClient(
        headers=headers,
    )


async def connect_mcp():
    """اتصال به MCP Server"""

    http_client = create_http_client()

    stream_context = streamable_http_client(
        MCP_SERVER_URL,
        http_client=http_client,
    )

    read_stream, write_stream = await stream_context.__aenter__()

    session = ClientSession(
        read_stream,
        write_stream,
    )

    await session.__aenter__()
    await session.initialize()

    return (
        http_client,
        stream_context,
        session,
    )


async def get_tools(
    session: ClientSession,
):
    """دریافت Toolهای MCP"""

    result = await session.list_tools()

    return result.tools


async def call_tool(
    session: ClientSession,
    tool_name: str,
    arguments: dict,
):
    """اجرای یک Tool در MCP"""

    result = await session.call_tool(
        tool_name,
        arguments,
    )

    return result


async def close_mcp(
    http_client: httpx.AsyncClient,
    stream_context,
    session: ClientSession,
) -> None:
    """بستن اتصال MCP"""

    await session.__aexit__(
        None,
        None,
        None,
    )

    await stream_context.__aexit__(
        None,
        None,
        None,
    )

    await http_client.aclose()