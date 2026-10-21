
import json
import os
from typing import Any, Final

from langchain_ollama import ChatOllama


MODEL_NAME: Final[str] = os.getenv(
    "DECISION_MODEL",
    "qwen3.5:4b",
).strip()

VALID_ROUTES: Final[set[str]] = {
    "chat",
    "tool",
    "rag",
    "tool_and_rag",
}


SYSTEM_PROMPT: Final[str] = """
تو ماژول تصمیم‌گیری یک دستیار هوشمند هستی.

وظیفه تو فقط تعیین مسیر درخواست است؛
نباید ابزارها را اجرا کنی، اطلاعاتی را از خودت بسازی
یا پاسخ نهایی درخواست کاربر را تولید کنی.

مسیرهای مجاز:

1. chat:
   درخواست گفت‌وگوی عادی است و به ابزار یا جست‌وجوی اسناد نیاز ندارد.

2. tool:
   برای پاسخ به درخواست، یک یا چند ابزار از فهرست موجود لازم است.

3. rag:
   درخواست نیازمند جست‌وجو در اسناد و منابع متنی موجود است.

4. tool_and_rag:
   درخواست هم به اجرای ابزار و هم به جست‌وجو در اسناد نیاز دارد.

قوانین:

- ابزارها را فقط از فهرست ارائه‌شده انتخاب کن.
- انتخاب ابزار باید بر اساس شرح و schema همان ابزار باشد.
- نام ابزارها و پارامترهای آن‌ها را از روی نام یا حدس انتخاب نکن.
- اگر چند ابزار برای بخش‌های مختلف درخواست لازم‌اند، همه را انتخاب کن.
- فقط پارامترهایی را ارسال کن که مقدارشان از درخواست کاربر
  یا از زمینه معتبر مکالمه قابل استخراج است.
- مقادیر متنی کاربر، مانند نام اشخاص، شرکت‌ها، برندها و تأمین‌کنندگان،
  باید بدون ترجمه یا تغییر معنایی حفظ شوند.
- اگر مقدار یک پارامتر مشخص نیست، آن را حدس نزن.
- برای پارامترهای اختیاریِ مشخص‌نشده، مقدار ارسال نکن.
- پارامترهای الزامی ابزار باید از اطلاعات موجود قابل تأمین باشند.
  اگر قابل تأمین نیستند، آن ابزار را فراخوانی نکن و به‌جای آن
  مسیر chat را انتخاب کن و در فیلد reason نیاز به اطلاعات بیشتر را توضیح بده.
- اگر درخواست به اسناد نیاز ندارد، rag_query را null قرار بده.
- اگر درخواست به ابزار نیاز ندارد، tool_calls باید آرایه خالی باشد.
- اگر درخواست به جست‌وجوی اسناد نیاز ندارد، rag_query باید null باشد.
- confidence عددی بین 0 و 1 است.
- خروجی باید فقط JSON معتبر باشد؛ بدون Markdown و توضیح بیرون از JSON.

قالب خروجی:

{
  "route": "chat | tool | rag | tool_and_rag",
  "tool_calls": [
    {
      "name": "نام دقیق ابزار",
      "arguments": {}
    }
  ],
  "rag_query": null,
  "confidence": 0.0,
  "reason": "دلیل کوتاه تصمیم"
}
""".strip()


def _tool_value(tool: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dictionary or an MCP tool object."""
    if isinstance(tool, dict):
        return tool.get(key, default)

    return getattr(tool, key, default)


def _normalize_tool(tool: Any) -> dict[str, Any]:
    """
    Convert tool metadata into a JSON-serializable dictionary.

    Supports dictionaries and MCP SDK objects.
    No tool names or tool-specific rules are hardcoded here.
    """
    name = _tool_value(tool, "name")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Every available tool must have a valid name.")

    description = _tool_value(tool, "description", "") or ""

    input_schema = _tool_value(tool, "input_schema")
    if input_schema is None:
        input_schema = _tool_value(tool, "inputSchema")

    if input_schema is None:
        input_schema = {
            "type": "object",
            "properties": {},
        }

    if not isinstance(input_schema, dict):
        if hasattr(input_schema, "model_dump"):
            input_schema = input_schema.model_dump()
        else:
            raise ValueError(
                f"Invalid input schema for tool '{name}'."
            )

    return {
        "name": name.strip(),
        "description": str(description),
        "input_schema": input_schema,
    }


def _validate_arguments(
    tool: dict[str, Any],
    arguments: dict[str, Any],
) -> None:
    """Validate required fields and basic JSON Schema types."""
    schema = tool["input_schema"]
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    if not isinstance(properties, dict):
        raise ValueError(
            f"Invalid properties schema for tool '{tool['name']}'."
        )

    if not isinstance(required, list):
        raise ValueError(
            f"Invalid required fields for tool '{tool['name']}'."
        )

    missing = [
        field
        for field in required
        if field not in arguments or arguments[field] is None
    ]

    if missing:
        raise ValueError(
            f"Missing required arguments for tool '{tool['name']}': "
            f"{', '.join(missing)}"
        )

    unknown = set(arguments) - set(properties)

    if unknown:
        raise ValueError(
            f"Unknown arguments for tool '{tool['name']}': "
            f"{', '.join(sorted(unknown))}"
        )

    type_checks = {
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int)
        and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float))
        and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
    }

    for argument_name, value in arguments.items():
        property_schema = properties[argument_name]

        if not isinstance(property_schema, dict):
            continue

        expected_type = property_schema.get("type")
        check = type_checks.get(expected_type)

        if check is not None and not check(value):
            raise ValueError(
                f"Argument '{argument_name}' for tool '{tool['name']}' "
                f"must have type '{expected_type}'."
            )

        if expected_type == "array":
            item_schema = property_schema.get("items", {})
            item_type = item_schema.get("type")

            item_check = type_checks.get(item_type)

            if item_check is not None:
                for index, item in enumerate(value):
                    if not item_check(item):
                        raise ValueError(
                            f"Item {index} of argument '{argument_name}' "
                            f"for tool '{tool['name']}' must have type "
                            f"'{item_type}'."
                        )


def _parse_model_output(raw_output: Any) -> dict[str, Any]:
    """Parse the model response as a JSON object."""
    if not isinstance(raw_output, str):
        # Supports response objects exposing a text field.
        content = getattr(raw_output, "content", None)

        if isinstance(content, str):
            raw_output = content
        else:
            raise ValueError(
                "The decision model did not return a JSON string."
            )

    try:
        decision = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The decision model returned invalid JSON."
        ) from exc

    if not isinstance(decision, dict):
        raise ValueError(
            "The decision model output must be a JSON object."
        )

    return decision


def _validate_decision(
    decision: dict[str, Any],
    available_tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the routing decision against the live tool catalog."""
    route = decision.get("route")

    if route not in VALID_ROUTES:
        raise ValueError(f"Unknown route: {route!r}")

    tool_calls = decision.get("tool_calls", [])
    rag_query = decision.get("rag_query")
    confidence = decision.get("confidence", 0.0)
    reason = decision.get("reason", "")

    if not isinstance(tool_calls, list):
        raise ValueError("'tool_calls' must be a list.")

    if rag_query is not None and not isinstance(rag_query, str):
        raise ValueError("'rag_query' must be a string or null.")

    if not isinstance(confidence, (int, float)) or isinstance(
        confidence, bool
    ):
        raise ValueError("'confidence' must be a number.")

    if not 0 <= confidence <= 1:
        raise ValueError("'confidence' must be between 0 and 1.")

    if not isinstance(reason, str):
        raise ValueError("'reason' must be a string.")

    tools_by_name = {
        tool["name"]: tool
        for tool in available_tools
    }

    validated_calls: list[dict[str, Any]] = []

    for call in tool_calls:
        if not isinstance(call, dict):
            raise ValueError("Each tool call must be a JSON object.")

        name = call.get("name")
        arguments = call.get("arguments", {})

        if not isinstance(name, str) or name not in tools_by_name:
            raise ValueError(
                f"Unknown tool selected by the model: {name!r}"
            )

        if not isinstance(arguments, dict):
            raise ValueError(
                f"Arguments for tool '{name}' must be an object."
            )

        _validate_arguments(tools_by_name[name], arguments)

        validated_calls.append(
            {
                "name": name,
                "arguments": arguments,
            }
        )

    has_tools = bool(validated_calls)
    has_rag = isinstance(rag_query, str) and bool(rag_query.strip())

    if route == "chat" and (has_tools or has_rag):
        raise ValueError(
            "The chat route cannot contain tool calls or a RAG query."
        )

    if route == "tool" and (not has_tools or has_rag):
        raise ValueError(
            "The tool route requires tool calls and no RAG query."
        )

    if route == "rag" and (has_tools or not has_rag):
        raise ValueError(
            "The rag route requires a RAG query and no tool calls."
        )

    if route == "tool_and_rag" and not (has_tools and has_rag):
        raise ValueError(
            "The tool_and_rag route requires tool calls and a RAG query."
        )

    return {
        "route": route,
        "tool_calls": validated_calls,
        "rag_query": rag_query.strip() if has_rag else None,
        "confidence": float(confidence),
        "reason": reason,
    }


def decide_route(
    query: str,
    available_tools: list[Any],
    model: Any | None = None,
) -> dict[str, Any]:
    """
    Decide how to handle a user request.

    This function only routes the request. It never executes tools.

    Args:
        query: User's message.
        available_tools: Tool metadata retrieved from MCP.
        model: Optional LangChain-compatible model, useful for testing.

    Returns:
        A validated routing decision.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if not isinstance(available_tools, list):
        raise ValueError("available_tools must be a list.")

    normalized_tools = [
        _normalize_tool(tool)
        for tool in available_tools
    ]

    tool_names = [tool["name"] for tool in normalized_tools]

    if len(tool_names) != len(set(tool_names)):
        raise ValueError("The available tool catalog contains duplicate names.")

    selected_model = model or ChatOllama(
                        model=MODEL_NAME,
                        temperature=0,
                        format="json",
                        reasoning=False,
                     )

    prompt = f"""
{SYSTEM_PROMPT}

فهرست ابزارهای موجود در MCP:
{json.dumps(normalized_tools, ensure_ascii=False, indent=2)}

درخواست کاربر:
{query}

تصمیم را مطابق قالب JSON مشخص‌شده برگردان.
""".strip()

    response = selected_model.invoke(prompt)

    print("\n===== DECISION MODEL DEBUG =====")
    print(f"Model: {MODEL_NAME}")
    print(f"Response type: {type(response).__name__}")
    print(f"Response repr: {response!r}")
    print("===== END DEBUG =====\n")

    decision = _parse_model_output(response)

    return _validate_decision(
        decision=decision,
        available_tools=normalized_tools,
    )
