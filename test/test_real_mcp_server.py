import asyncio

from module.mcp_module import (
    connect_mcp,
    close_mcp,
)
from module.tool_module import (
    get_available_tools,
    execute_tool,
)


EXPECTED_TOOLS: list[str] = [
    "get_top_purchased_products",
    "get_products_purchase_orders",
    "get_vendor_purchase_share",
    "get_contract_purchase_order_price_mismatch",
    "get_upcoming_purchase_orders",
    "get_vendors_with_highest_delivery_delay",
]

TEST_TOOL_NAME = "get_vendors_with_highest_delivery_delay"


async def run_test() -> None:
    """تست واقعی کشف و اجرای ابزارهای MCP Server"""

    http_client = None
    stream_context = None
    session = None

    try:
        (
            http_client,
            stream_context,
            session,
        ) = await connect_mcp()

        # دریافت ابزارهای واقعی سرور
        tools = await get_available_tools(
            session=session,
        )

        actual_tool_names: list[str] = [
            tool.name
            for tool in tools
        ]

        print()
        print("===== AVAILABLE MCP TOOLS =====")

        for tool in tools:
            print(f"\nName: {tool.name}")
            print(f"Description: {tool.description}")
            print(f"Input Schema: {tool.input_schema}")

        # بررسی وجود شش ابزار مورد انتظار
        missing_tools: list[str] = [
            name
            for name in EXPECTED_TOOLS
            if name not in actual_tool_names
        ]

        assert not missing_tools, (
            f"Missing MCP tools: {missing_tools}\n"
            f"Available tools: {actual_tool_names}"
        )

        print()
        print("All expected ERP tools were discovered.")

        # انتخاب یک ابزار گزارش‌گیری واقعی
        selected_tool = next(
            (
                tool
                for tool in tools
                if tool.name == TEST_TOOL_NAME
            ),
            None,
        )

        assert selected_tool is not None

        # اجرای ابزار با تعداد محدود نتایج
        arguments: dict[str, object] = {
            "topCount": 3,
        }

        print()
        print("===== EXECUTING MCP TOOL =====")
        print(f"Tool: {selected_tool.name}")
        print(f"Arguments: {arguments}")

        result = await execute_tool(
            session=session,
            tool_name=selected_tool.name,
            arguments=arguments,
        )

        print()
        print("===== TOOL RESULT =====")
        print(result)

        assert result is not None

        if getattr(result, "isError", False):
            raise AssertionError(
                f"MCP tool returned an error: {result}"
            )

        print()
        print("test_real_mcp_server: PASSED")

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


def test_real_mcp_server() -> None:
    """اجرای تست با pytest"""

    asyncio.run(
        run_test(),
    )


if __name__ == "__main__":
    asyncio.run(
        run_test(),
    )
