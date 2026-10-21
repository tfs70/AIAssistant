
import asyncio
import json

from module.decision_module import decide_route
from module.mcp_module import connect_mcp, close_mcp
from module.tool_module import get_available_tools


async def main() -> None:
    http_client, stream_context, session = await connect_mcp()

    try:
        tools = await get_available_tools(session)

        print(f"\nAvailable MCP tools: {len(tools)}")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

        questions = [
            "سلام، حالت چطوره؟",
            "کدام کالاها واقعاً بیشترین خرید را داشته‌اند؟",
        ]

        for question in questions:
            print(f"\n{'=' * 60}")
            print(f"Question: {question}")

            decision = decide_route(
                query=question,
                available_tools=tools,
            )

            print(
                json.dumps(
                    decision,
                    ensure_ascii=False,
                    indent=2,
                )
            )

    finally:
        await close_mcp(
            http_client,
            stream_context,
            session,
        )


if __name__ == "__main__":
    asyncio.run(main())
