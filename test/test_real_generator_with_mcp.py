
import asyncio
import json

from module.decision_module import decide_tool
from module.mcp_module import connect_mcp, close_mcp
from module.tool_module import get_available_tools, execute_tool


QUESTION = "سه تأمین‌کننده با بیشترین مجموع روزهای تأخیر را بده"


def print_json(title: str, data: dict) -> None:
    print(f"\n===== {title} =====")
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def print_tool_result(result) -> None:
    print("\n===== MCP TOOL RESULT =====")

    is_error = getattr(result, "isError", False)
    print(f"Is error: {is_error}")

    content = getattr(result, "content", None)

    if content:
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                print(text)
            else:
                print(item)
    else:
        print(result)


async def main() -> None:
    http_client = None
    stream_context = None
    session = None

    try:
        # 1. Connect to the existing authenticated MCP server.
        http_client, stream_context, session = await connect_mcp()

        print("MCP connection established.")

        # 2. Discover actual tools exposed by the server.
        tools = await get_available_tools(session=session)
        available_tool_names = {tool.name for tool in tools}

        print("\n===== AVAILABLE TOOLS =====")
        for tool in tools:
            print(f"- {tool.name}")

        # 3. Ask the decision model to select a tool.
        decision = decide_tool(query=QUESTION)

        print_json("DECISION", decision)

        # 4. Stop safely if the decision output is invalid.
        if decision.get("error"):
            print("\nDecision is invalid. MCP execution skipped.")
            return

        # 5. A valid decision may indicate that no tool is required.
        if decision.get("requires_tool") is not True:
            print("\nNo MCP tool is required for this question.")
            return

        tool_name = decision.get("tool_name")
        arguments = decision.get("arguments")

        # 6. Validate the selected tool against the real server tools.
        if not isinstance(tool_name, str) or tool_name not in available_tool_names:
            print(f"\nUnknown MCP tool: {tool_name!r}")
            print("MCP execution skipped.")
            return

        if not isinstance(arguments, dict):
            print("\nInvalid arguments. MCP execution skipped.")
            return

        # 7. Execute only the validated tool.
        print(f"\nExecuting MCP tool: {tool_name}")
        print_json("TOOL ARGUMENTS", arguments)

        result = await execute_tool(
            session=session,
            tool_name=tool_name,
            arguments=arguments,
        )

        print_tool_result(result)

    except Exception as exc:
        print(f"\nTEST FAILED: {type(exc).__name__}: {exc}")
        raise

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
    asyncio.run(main())
