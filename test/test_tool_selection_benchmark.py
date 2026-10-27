
import asyncio
import os
import time
from typing import Any

from langchain_ollama import ChatOllama

from module.decision_module import decide_route
from module.mcp_module import connect_mcp, close_mcp
from module.tool_module import get_available_tools


QUESTIONS: list[dict[str, Any]] = [
    # خرید واقعی
    {
        "query": "بیشترین کالاهایی که خریدیم رو نشون بده",
        "tool": "get_top_purchased_products",
        "args": {},
    },
    {
        "query": "سه کالایی که بیشترین مقدار خرید واقعی رو داشتن کدوما هستن؟",
        "tool": "get_top_purchased_products",
        "args": {"topCount": 3},
    },
    {
        "query": "کالاهای خریداری‌شده از برند بهار رو نمایش بده",
        "tool": "get_top_purchased_products",
        "args": {"brandNames": ["بهار"]},
    },

    # سفارش‌های خرید
    {
        "query": "بیشترین کالاهایی که سفارش دادیم کدوما هستن؟",
        "tool": "get_products_purchase_orders",
        "args": {},
    },
    {
        "query": "پنج کالای اول در سفارش‌های خرید رو بده",
        "tool": "get_products_purchase_orders",
        "args": {"topCount": 5},
    },
    {
        "query": "سفارش‌های خرید برند بهار از شرکت پارس رو بررسی کن",
        "tool": "get_products_purchase_orders",
        "args": {
            "brandNames": ["بهار"],
            "vendorNames": ["پارس"],
        },
    },

    # سهم تأمین‌کننده
    {
        "query": "سهم شرکت پارس از کل خریدهای ما چقدره؟",
        "tool": "get_vendor_purchase_share",
        "args": {"vendorNames": ["پارس"]},
    },
    {
        "query": "درصد خرید از تأمین‌کننده آریا رو بگو",
        "tool": "get_vendor_purchase_share",
        "args": {"vendorNames": ["آریا"]},
    },
    {
        "query": "شرکت تاک آریا تجارت چند درصد از خریدهای ما رو تأمین کرده؟",
        "tool": "get_vendor_purchase_share",
        "args": {"vendorNames": ["صنایع غذایی بهار"]},
    },

    # مغایرت قیمت
    {
        "query": "کالاهایی که قیمت قراردادشون با سفارش خرید فرق داره رو پیدا کن",
        "tool": "get_contract_purchase_order_price_mismatch",
        "args": {},
    },
    {
        "query": "ده مورد مغایرت قیمت قرارداد و سفارش خرید رو نشون بده",
        "tool": "get_contract_purchase_order_price_mismatch",
        "args": {"topCount": 10},
    },
    {
        "query": "مغایرت قیمت سفارش و قرارداد برای برند بهار چیه؟",
        "tool": "get_contract_purchase_order_price_mismatch",
        "args": {"brandNames": ["بهار"]},
    },

    # سفارش‌های پیش رو
    {
        "query": "چه سفارش‌های خریدی طی روزهای آینده تاریخ حمل دارن؟",
        "tool": "get_upcoming_purchase_orders",
        "args": {},
    },
    {
        "query": "سفارش‌های خریدی که تا ۷ روز آینده حمل می‌شن رو نشون بده",
        "tool": "get_upcoming_purchase_orders",
        "args": {"daysAhead": 7},
    },
    {
        "query": "دو سفارش خرید با تاریخ حمل در سه روز آینده رو نمایش بده",
        "tool": "get_upcoming_purchase_orders",
        "args": {"topCount": 2, "daysAhead": 3},
    },

    # تأخیر تأمین‌کنندگان
    {
        "query": "کدوم تأمین‌کننده‌ها بیشترین تأخیر در تحویل رو داشتن؟",
        "tool": "get_vendors_with_highest_delivery_delay",
        "args": {},
    },
    {
        "query": "سه تأمین‌کننده با بیشترین مجموع روزهای تأخیر رو بده",
        "tool": "get_vendors_with_highest_delivery_delay",
        "args": {"topCount": 3},
    },
    {
        "query": "بیشترین تأخیر در رسیدهای خرید مربوط به کدوم فروشنده‌هاست؟",
        "tool": "get_vendors_with_highest_delivery_delay",
        "args": {},
    },

    # سؤال‌هایی که نباید ابزار ERP را فعال کنند
    {
        "query": "سلام، حالت چطوره؟",
        "tool": None,
        "args": {},
    },
    {
        "query": "فرق سفارش خرید با خرید واقعی چیه؟",
        "tool": None,
        "args": {},
    },
    {
        "query": "برای گزارش تأخیر تأمین‌کننده چه شاخص‌هایی مفیدن؟",
        "tool": None,
        "args": {},
    },
    {
        "query": "اسم من علی است",
        "tool": None,
        "args": {},
    },
    {
        "query": "لطفاً مفهوم سهم تأمین‌کننده رو توضیح بده",
        "tool": None,
        "args": {},
    },
    {
        "query": "برای تحلیل فروش چه نمودارهایی پیشنهاد می‌کنی؟",
        "tool": None,
        "args": {},
    },
]


def _evaluate_result(
    result: dict[str, Any],
    expected_tool: str | None,
    expected_args: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a routing result against one benchmark case."""

    route = result.get("route")
    tool_calls = result.get("tool_calls", [])

    if not isinstance(tool_calls, list):
        raise ValueError("'tool_calls' must be a list.")

    actual_tool = None
    actual_args: dict[str, Any] = {}

    if tool_calls:
        first_call = tool_calls[0]

        if not isinstance(first_call, dict):
            raise ValueError("Each tool call must be an object.")

        actual_tool = first_call.get("name")
        actual_args = first_call.get("arguments", {})

        if not isinstance(actual_args, dict):
            raise ValueError("Tool arguments must be an object.")

    if expected_tool is None:
        # For non-tool questions, chat and RAG are both acceptable.
        decision_ok = (
            route in {"chat", "rag"}
            and not tool_calls
            and result.get("rag_query") is not None
            if route == "rag"
            else route == "chat" and not tool_calls
        )
        tool_ok = not tool_calls
        args_ok = tool_ok and actual_args == expected_args

    else:
        decision_ok = (
            route in {"tool", "tool_and_rag"}
            and len(tool_calls) == 1
            and actual_tool == expected_tool
        )
        tool_ok = actual_tool == expected_tool
        args_ok = (
            tool_ok
            and len(tool_calls) == 1
            and actual_args == expected_args
        )

    return {
        "decision_ok": decision_ok,
        "tool_ok": tool_ok,
        "args_ok": args_ok,
        "actual_tool": actual_tool,
        "actual_args": actual_args,
    }


async def run_benchmark() -> None:
    model_name = os.getenv(
        "DECISION_MODEL",
        "qwen3.5:4b",
    ).strip()

    model = ChatOllama(
        model=model_name,
        temperature=0,
        format="json",
        reasoning=False,
    )

    http_client = None
    stream_context = None
    session = None

    try:
        http_client, stream_context, session = await connect_mcp()

        available_tools = await get_available_tools(session=session)

        if not available_tools:
            raise RuntimeError(
                "MCP returned an empty tool catalog."
            )

        print(f"\nModel: {model_name}")
        print(f"Available MCP tools: {len(available_tools)}")
        print(f"Questions: {len(QUESTIONS)}")
        print("=" * 80)

        total = len(QUESTIONS)
        decision_correct = 0
        tool_correct = 0
        args_correct = 0
        errors = 0
        total_seconds = 0.0

        for index, case in enumerate(QUESTIONS, start=1):
            started_at = time.perf_counter()
            error_message = None
            result: dict[str, Any] = {}

            try:
                result = decide_route(
                    query=case["query"],
                    available_tools=available_tools,
                    model=model,
                )

                evaluation = _evaluate_result(
                    result=result,
                    expected_tool=case["tool"],
                    expected_args=case["args"],
                )

                decision_ok = evaluation["decision_ok"]
                tool_ok = evaluation["tool_ok"]
                args_ok = evaluation["args_ok"]

            except Exception as exc:
                error_message = repr(exc)
                decision_ok = False
                tool_ok = False
                args_ok = False
                evaluation = {
                    "actual_tool": None,
                    "actual_args": {},
                }

            elapsed = time.perf_counter() - started_at
            total_seconds += elapsed

            decision_correct += int(decision_ok)
            tool_correct += int(tool_ok)
            args_correct += int(args_ok)
            errors += int(error_message is not None)

            status = (
                "PASS"
                if decision_ok and args_ok and error_message is None
                else "FAIL"
            )

            print(f"\n[{index:02}/{total}] {status}")
            print(f"Question:  {case['query']}")
            print(f"Expected route/tool: {case['tool']}")
            print(f"Actual route:        {result.get('route')}")
            print(f"Actual tool:         {evaluation['actual_tool']}")
            print(f"Expected arguments:  {case['args']}")
            print(f"Actual arguments:    {evaluation['actual_args']}")
            print(f"Confidence:          {result.get('confidence', 0.0)}")
            print(f"Latency:             {elapsed:.2f}s")

            if error_message:
                print(f"Error: {error_message}")
                continue

            if not args_ok:
                print(f"Reason: {result.get('reason', '')}")

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print(f"Model: {model_name}")
        print(
            f"Decision accuracy: {decision_correct}/{total} "
            f"({decision_correct / total:.1%})"
        )
        print(
            f"Tool accuracy:     {tool_correct}/{total} "
            f"({tool_correct / total:.1%})"
        )
        print(
            f"Exact args:        {args_correct}/{total} "
            f"({args_correct / total:.1%})"
        )
        print(f"Invalid outputs:   {errors}/{total}")
        print(f"Total time:        {total_seconds:.2f}s")
        print(f"Average latency:   {total_seconds / total:.2f}s")

    finally:
        if (
            http_client is not None
            and stream_context is not None
            and session is not None
        ):
            await close_mcp(
                http_client,
                stream_context,
                session,
            )


if __name__ == "__main__":
    asyncio.run(run_benchmark())
