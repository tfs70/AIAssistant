
import asyncio
import json

from module.mcp_tool_executor_module import route_and_execute_question


QUESTION = "سه تأمین‌کننده با بیشترین مجموع روزهای تأخیر را بده"


async def main() -> None:
    result = await route_and_execute_question(QUESTION)

    if not isinstance(result, dict):
        raise TypeError(
            "route_and_execute_question must return a dictionary."
        )

    print("\n===== MCP ROUTING RESULT =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    route = result.get("route")
    tool_results = result.get("tool_results")

    if route not in {"chat", "tool", "rag", "tool_and_rag"}:
        raise AssertionError(f"Unexpected route: {route!r}")

    if not isinstance(tool_results, list):
        raise AssertionError(
            "'tool_results' must be a list."
        )

    if route == "tool" and not tool_results:
        raise AssertionError(
            "The tool route returned no tool results."
        )

    print("\n===== TEST COMPLETED =====")


if __name__ == "__main__":
    asyncio.run(main())
