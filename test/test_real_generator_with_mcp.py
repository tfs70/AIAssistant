import asyncio

from module.mcp_module import (
    connect_mcp,
    close_mcp,
)
from module.tool_module import (
    get_available_tools,
    execute_tool,
)


TOOL_NAME = "get_random_number"


async def test_real_generator_with_mcp() -> None:
    """تست واقعی اجرای Tool از طریق MCP"""

    http_client = None
    stream_context = None
    session = None

    try:
        # -------------------------------------------------
        # MCP Connection
        # -------------------------------------------------

        (
            http_client,
            stream_context,
            session,
        ) = await connect_mcp()

        print()
        print("===== MCP TOOLS =====")

        tools = await get_available_tools(
            session=session,
        )

        assert tools

        for tool in tools:
            print(
                f"- {tool.name}",
            )

        # -------------------------------------------------
        # Find Tool
        # -------------------------------------------------

        selected_tool = next(
            (
                tool
                for tool in tools
                if tool.name == TOOL_NAME
            ),
            None,
        )

        assert selected_tool is not None, (
            f"Tool '{TOOL_NAME}' not found"
        )

        print()
        print("===== SELECTED TOOL =====")
        print(f"Name: {selected_tool.name}")
        print(f"Description: {selected_tool.description}")

        # -------------------------------------------------
        # Tool Arguments
        # -------------------------------------------------

        print()
        print("===== INPUT SCHEMA =====")
        print(selected_tool.input_schema)

        # -------------------------------------------------
        # Execute Tool
        # -------------------------------------------------

        # این Arguments را بر اساس Tool واقعی خودت تغییر بده.
        arguments: dict = {}

        result = await execute_tool(
            session=session,
            tool_name=selected_tool.name,
            arguments=arguments,
        )

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        print()
        print("===== TOOL RESULT =====")
        print(result)

        assert result is not None

        print()
        print("test_real_generator_with_mcp: PASSED")

    finally:
        if (
            http_client is not None
            and stream_context is not None
            and session is not None
        ):
            await close_mcp(
                http_client=http_client,
                stream_context=stream_context,
                session=session,
            )


if __name__ == "__main__":
    asyncio.run(
        test_real_generator_with_mcp(),
    )