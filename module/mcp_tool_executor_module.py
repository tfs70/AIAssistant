
import asyncio
import json
from typing import Any

from mcp import ClientSession

from module.decision_module import decide_tool
from module.mcp_module import (
    close_mcp,
    connect_mcp,
)
from module.tool_module import (
    execute_tool,
    get_available_tools,
)


ERP_TOOL_NAMES = {
    "get_top_purchased_products",
    "get_products_purchase_orders",
    "get_vendor_purchase_share",
    "get_contract_purchase_order_price_mismatch",
    "get_upcoming_purchase_orders",
    "get_vendors_with_highest_delivery_delay",
}


def _validate_decision(
    decision: dict[str, Any],
    available_tools: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Validate a tool decision against the actual MCP schema."""

    if decision.get("error"):
        raise ValueError(
            f"Invalid tool decision: {decision['error']}"
        )

    if decision.get("requires_tool") is not True:
        return None

    tool_name = decision.get("tool_name")
    arguments = decision.get("arguments")

    if not isinstance(tool_name, str):
        raise ValueError("Tool name is missing.")

    if tool_name not in ERP_TOOL_NAMES:
        raise ValueError(
            f"Tool is not an allowed ERP tool: {tool_name}"
        )

    if tool_name not in available_tools:
        raise ValueError(
            f"Tool is not available on MCP server: {tool_name}"
        )

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a dictionary.")


    tool = available_tools[tool_name]
    schema = getattr(tool, "input_schema", {}) or {}

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    unknown_arguments = set(arguments) - set(properties)

    if unknown_arguments:
        raise ValueError(
            f"Unknown arguments for {tool_name}: "
            f"{sorted(unknown_arguments)}"
        )

    missing_arguments = [
        name
        for name in required
        if name not in arguments
    ]

    if missing_arguments:
        raise ValueError(
            f"Missing required arguments for {tool_name}: "
            f"{missing_arguments}"
        )

    unknown_arguments = set(arguments) - set(properties)
    if unknown_arguments:
        raise ValueError(
            f"Unknown arguments for {tool_name}: "
            f"{sorted(unknown_arguments)}"
        )

    missing_arguments = [
        name for name in required
        if name not in arguments
    ]
    if missing_arguments:
        raise ValueError(
            f"Missing required arguments for {tool_name}: "
            f"{missing_arguments}"
        )

    if (
        tool_name == "get_vendor_purchase_share"
        and not arguments.get("vendorNames")
    ):
        raise ValueError(
            "vendorNames is required for get_vendor_purchase_share."
        )

    return tool_name, arguments

def _format_tool_result(result: Any) -> str:
    """Convert MCP content into readable text for the generator."""

    if getattr(result, "isError", False):
        raise RuntimeError("MCP server returned a tool execution error.")

    content = getattr(result, "content", None)

    if not content:
        return str(result)

    parts: list[str] = []

    for item in content:
        item_text = getattr(item, "text", None)

        if item_text is None:
            parts.append(str(item))
            continue

        try:
            parsed = json.loads(item_text)
            item_text = json.dumps(
                parsed,
                ensure_ascii=False,
                indent=2,
            )
        except (json.JSONDecodeError, TypeError):
            pass

        parts.append(item_text)

    return "\n".join(parts)


async def _execute_mcp_for_question(
    question: str,
) -> str | None:
    """Decide whether an ERP tool is needed and execute it if required."""

    # Step 1: Decide before connecting to MCP.

    # پیام‌های عادی نباید مدل تصمیم‌گیری ERP را اجرا کنند.
    if not _may_need_erp_tool(question):
        return None

    # فقط سؤال‌های محتمل ERP وارد مدل تصمیم‌گیری می‌شوند.
    decision = decide_tool(query=question)

    if decision.get("error"):
        raise ValueError(
            f"Invalid tool decision: {decision['error']}"
        )

    # No MCP connection is needed for ordinary questions.
    if decision.get("requires_tool") is not True:
        return None

    http_client = None
    stream_context = None
    session: ClientSession | None = None

    try:
        # Step 2: Connect only when an ERP tool is required.
        http_client, stream_context, session = await connect_mcp()

        tools = await get_available_tools(session=session)

        available_tools = {
            tool.name: tool
            for tool in tools
        }

        # Step 3: Validate the decision against actual MCP tools.
        validated = _validate_decision(
            decision=decision,
            available_tools=available_tools,
        )

        if validated is None:
            return None

        tool_name, arguments = validated

        # Step 4: Execute the selected tool.
        result = await execute_tool(
            session=session,
            tool_name=tool_name,
            arguments=arguments,
        )

        # Step 5: Return the real result for the final answer.
        return (
            f"Tool: {tool_name}\n"
            f"Arguments: "
            f"{json.dumps(arguments, ensure_ascii=False)}\n"
            f"Result:\n{_format_tool_result(result)}"
        )

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
            

def execute_mcp_for_question(
    question: str,
) -> str | None:
    """
    Synchronous adapter for the current synchronous conversation module.

    Returns None when no tool is required.
    Returns formatted ERP data when a tool succeeds.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _execute_mcp_for_question(question)
        )

    raise RuntimeError(
        "execute_mcp_for_question cannot run inside an active "
        "event loop. Use the async implementation directly."
    )


def _may_need_erp_tool(question: str) -> bool:
    """بررسی اولیه برای حذف پیام‌های آشکارا غیر ERP."""

    normalized = question.casefold()

    erp_keywords = (
        "خرید",
        "خریداری",
        "سفارش",
        "تأمین",
        "تامین",
        "تأمین‌کننده",
        "تامین کننده",
        "فروشنده",
        "کالا",
        "محصول",
        "برند",
        "قرارداد",
        "مغایرت",
        "تأخیر",
        "تاخیر",
        "رسید خرید",
        "روزهای تأخیر",
        "روزهای تاخیر",
        "سهم خرید",
        "درصد خرید",
        "حمل",
        "erp",
        "purchase order",
        "supplier",
        "vendor",
    )

    return any(keyword in normalized for keyword in erp_keywords)
