
import asyncio
import json

from module.mcp_module import connect_mcp, close_mcp
from module.tool_module import get_available_tools


async def main():
    http_client, stream_context, session = await connect_mcp()

    try:
        tools = await get_available_tools(session)

        for tool in tools:
            print(f"\n{'=' * 70}")
            print(f"Tool: {tool.name}")

            if hasattr(tool, "model_dump"):
                print("Full MCP tool object:")
                print(json.dumps(
                    tool.model_dump(by_alias=True, exclude_none=False),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ))
            else:
                print("Tool attributes:")
                print(vars(tool))

            print("inputSchema:", getattr(tool, "inputSchema", "<missing>"))
            print("input_schema:", getattr(tool, "input_schema", "<missing>"))
    finally:
        await close_mcp(
            http_client,
            stream_context,
            session,
        )


if __name__ == "__main__":
    asyncio.run(main())
