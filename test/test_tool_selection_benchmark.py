
import json
import os
import time
from typing import Any

from module.decision_module import decide_tool


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
        "query": "شرکت صنایع غذایی بهار چند درصد از خریدهای ما رو تأمین کرده؟",
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


def run_benchmark() -> None:
    model_name = os.getenv(
        "DECISION_MODEL",
        "qwen3.5:4b",
    )

    total = len(QUESTIONS)
    tool_correct = 0
    decision_correct = 0
    args_correct = 0
    errors = 0
    total_seconds = 0.0

    print(f"\nModel: {model_name}")
    print(f"Questions: {total}")
    print("=" * 80)

    for index, case in enumerate(QUESTIONS, start=1):
        started_at = time.perf_counter()

        try:
            result = decide_tool(
                query=case["query"],
                model_name=model_name,
            )
        except Exception as exc:
            result = {
                "requires_tool": False,
                "tool_name": None,
                "arguments": {},
                "error": repr(exc),
            }

        elapsed = time.perf_counter() - started_at
        total_seconds += elapsed

        expected_tool = case["tool"]
        expected_args = case["args"]

        actual_tool = result.get("tool_name")
        actual_args = result.get("arguments", {})
        has_error = bool(result.get("error"))

        expected_requires_tool = expected_tool is not None
        actual_requires_tool = result.get("requires_tool", False)

        decision_ok = (
            actual_requires_tool == expected_requires_tool
            and actual_tool == expected_tool
        )
        tool_ok = actual_tool == expected_tool
        args_ok = tool_ok and actual_args == expected_args

        decision_correct += int(decision_ok)
        tool_correct += int(tool_ok)
        args_correct += int(args_ok)
        errors += int(has_error)

        status = (
            "PASS"
            if decision_ok and args_ok and not has_error
            else "FAIL"
        )

        print(f"\n[{index:02}/{total}] {status}")
        print(f"Question:  {case['query']}")
        print(f"Expected:  {expected_tool}")
        print(f"Actual:    {actual_tool}")
        print(f"Expected arguments: {expected_args}")
        print(f"Actual arguments:   {actual_args}")
        print(f"Confidence: {result.get('confidence', 0.0)}")
        print(f"Latency:    {elapsed:.2f}s")

        if has_error:
            print(f"Error: {result['error']}")
            if "raw_response" in result:
                print(f"Raw response: {result['raw_response']}")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print(f"Model: {model_name}")
    print(f"Decision accuracy: {decision_correct}/{total} "
          f"({decision_correct / total:.1%})")
    print(f"Tool accuracy:     {tool_correct}/{total} "
          f"({tool_correct / total:.1%})")
    print(f"Exact args:        {args_correct}/{total} "
          f"({args_correct / total:.1%})")
    print(f"Invalid outputs:   {errors}/{total}")
    print(f"Total time:        {total_seconds:.2f}s")
    print(f"Average latency:   {total_seconds / total:.2f}s")


if __name__ == "__main__":
    run_benchmark()
