import json
from unittest.mock import Mock

import pytest

from module.decision_module import decide_route


TOOLS = [
    {
        "name": "GetActualPurchases",
        "description": "Returns products that were actually purchased.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topCount": {"type": "integer"},
                "vendorNames": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [],
        },
    },
    {
        "name": "GetPurchaseOrders",
        "description": "Returns products in purchase orders, not actual purchases.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topCount": {"type": "integer"},
            },
            "required": [],
        },
    },
]


def make_model_response(decision: dict) -> Mock:
    model = Mock()
    model.invoke.return_value = json.dumps(
        decision,
        ensure_ascii=False,
    )
    return model


def test_greeting_routes_to_chat():
    model = make_model_response({
        "route": "chat",
        "tool_calls": [],
        "rag_query": None,
        "confidence": 0.98,
    })

    result = decide_route(
        query="سلام، خوبی؟",
        available_tools=TOOLS,
        model=model,
    )

    assert result["route"] == "chat"
    assert result["tool_calls"] == []


def test_actual_purchases_selects_matching_tool():
    model = make_model_response({
        "route": "tool",
        "tool_calls": [
            {
                "name": "GetActualPurchases",
                "arguments": {"topCount": 10},
            }
        ],
        "rag_query": None,
        "confidence": 0.95,
    })

    result = decide_route(
        query="ده محصولی که بیشترین خرید واقعی را داشته‌اند نشان بده",
        available_tools=TOOLS,
        model=model,
    )

    assert result["route"] == "tool"
    assert result["tool_calls"][0]["name"] == "GetActualPurchases"


def test_purchase_orders_are_distinguished_from_actual_purchases():
    model = make_model_response({
        "route": "tool",
        "tool_calls": [
            {
                "name": "GetPurchaseOrders",
                "arguments": {"topCount": 10},
            }
        ],
        "rag_query": None,
        "confidence": 0.94,
    })

    result = decide_route(
        query="ده محصول اول در سفارش‌های خرید را نشان بده",
        available_tools=TOOLS,
        model=model,
    )

    assert result["tool_calls"][0]["name"] == "GetPurchaseOrders"


def test_document_question_routes_to_rag():
    model = make_model_response({
        "route": "rag",
        "tool_calls": [],
        "rag_query": "شرایط و مراحل فرایند خرید",
        "confidence": 0.91,
    })

    result = decide_route(
        query="شرایط و مراحل فرایند خرید در مستندات چیست؟",
        available_tools=TOOLS,
        model=model,
    )

    assert result["route"] == "rag"
    assert result["rag_query"]


def test_combined_request_routes_to_tool_and_rag():
    model = make_model_response({
        "route": "tool_and_rag",
        "tool_calls": [
            {
                "name": "GetActualPurchases",
                "arguments": {"topCount": 5},
            }
        ],
        "rag_query": "تعریف و معیارهای خرید واقعی",
        "confidence": 0.89,
    })

    result = decide_route(
        query=(
            "پنج محصولی را که بیشترین خرید واقعی داشته‌اند پیدا کن "
            "و معیار تعریف خرید واقعی در مستندات را هم توضیح بده."
        ),
        available_tools=TOOLS,
        model=model,
    )

    assert result["route"] == "tool_and_rag"
    assert result["tool_calls"][0]["name"] == "GetActualPurchases"
    assert result["rag_query"]


def test_router_prompt_uses_dynamic_tool_catalog():
    model = make_model_response({
        "route": "tool",
        "tool_calls": [
            {
                "name": "GetActualPurchases",
                "arguments": {},
            }
        ],
        "rag_query": None,
        "confidence": 0.90,
    })

    decide_route(
        query="محصولات خریداری‌شده را نشان بده",
        available_tools=TOOLS,
        model=model,
    )

    prompt = str(model.invoke.call_args)

    assert "GetActualPurchases" in prompt
    assert "GetPurchaseOrders" in prompt
    assert "vendorNames" in prompt


def test_router_does_not_accept_unknown_tool():
    model = make_model_response({
        "route": "tool",
        "tool_calls": [
            {
                "name": "HardcodedOrUnknownTool",
                "arguments": {},
            }
        ],
        "rag_query": None,
        "confidence": 0.99,
    })

    with pytest.raises((ValueError, KeyError)):
        decide_route(
            query="اطلاعات خرید را بده",
            available_tools=TOOLS,
            model=model,
        )


def test_router_rejects_malformed_json():
    model = Mock()
    model.invoke.return_value = "این پاسخ JSON معتبر نیست"

    with pytest.raises((ValueError, json.JSONDecodeError)):
        decide_route(
            query="اطلاعات خرید را بده",
            available_tools=TOOLS,
            model=model,
        )