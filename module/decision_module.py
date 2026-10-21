
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
نباید ابزارها را اجرا کنی یا پاسخ نهایی کاربر را تولید کنی.

مسیرهای مجاز:
- chat: گفت‌وگوی عادی، بدون ابزار و بدون جست‌وجوی اسناد
- tool: استفاده از یک یا چند ابزار
- rag: جست‌وجو در اسناد
- tool_and_rag: استفاده هم‌زمان از ابزار و جست‌وجوی اسناد

قوانین انتخاب ابزار:
1. ابزارها را فقط از فهرست ارائه‌شده انتخاب کن.
2. انتخاب را بر اساس توضیحات و input_schema واقعی ابزار انجام بده.
3. اگر چند ابزار برای بخش‌های مختلف درخواست لازم‌اند، همه را انتخاب کن.
4. اگر ابزار لازم نیست، tool_calls باید آرایه خالی باشد.
5. نام ابزار و پارامترها را حدس نزن.


قوانین بسیار مهم آرگومان‌ها:

1. ابتدا منظور کاربر را از درخواست فعلی و زمینه معتبر مکالمه استخراج کن؛
   سپس ابزار مناسب را انتخاب کن و آرگومان‌های آن را بساز.

2. اصل اساسی: حداقل آرگومان‌های لازم را ارسال کن.
   هر آرگومان باید شواهد مشخصی در درخواست فعلی یا زمینه معتبر مکالمه
   داشته باشد. اگر چنین شواهدی وجود ندارد، آرگومان را حذف کن؛
   حتی اگر مقدار آن متداول، مفید یا منطقی به نظر برسد.

3. مقدار پارامتر را فقط در یکی از این شرایط ارسال کن:
   - کاربر مقدار آن را صریحاً بیان کرده است.
   - مقدار آن بدون ابهام از درخواست کاربر قابل استخراج است.
   - مقدار آن در زمینه معتبر مکالمه مشخص شده و هنوز برای درخواست
     فعلی معتبر است.

4. اطلاعات عمومی، عادت‌های رایج، نمونه‌های موجود در توضیحات ابزار،
   مقادیر پیش‌فرض schema و حدس شخصی، شواهد معتبر برای تعیین مقدار
   پارامتر نیستند.

5. اگر کاربر تعداد نتایج را مشخص نکرده است، پارامتر مربوط به تعداد
   نتایج را ارسال نکن. واژه‌هایی مانند «بیشترین»، «برترین» یا
   «لیست کن» به‌تنهایی تعداد مشخصی را تعیین نمی‌کنند.

6. اگر کاربر بازه زمانی مشخصی را بیان کرده است، مقدار پارامتر زمانی
   را دقیقاً از همان بازه استخراج کن؛ اما پارامترهای اختیاری نامرتبط
   را صرفاً به دلیل مشخص بودن بازه زمانی اضافه نکن.

7. مقدار default در input_schema متعلق به قرارداد ابزار است.
   اگر کاربر مقدار آن پارامتر را مشخص نکرده، پارامتر اختیاری را
   حذف کن تا ابزار یا سرور MCP مطابق قرارداد خودش عمل کند.

8. پارامتر نامشخص را با null، صفر، عدد رایج یا مقدار پیش‌فرض
   جایگزین نکن؛ آن را از arguments حذف کن، مگر اینکه پارامتر
   اجباری باشد و schema یا قرارداد ابزار صراحتاً مقدار null را
   به‌عنوان ورودی معتبر الزامی کرده باشد.

9. نام اشخاص، شرکت‌ها، برندها و تأمین‌کنندگان را با حفظ عبارت
   ارائه‌شده توسط کاربر منتقل کن. پیشوندهای نام شرکت را خودسرانه
   حذف نکن و نام را ترجمه یا بازنویسی نکن.

10. قبل از تولید JSON نهایی، تمام آرگومان‌ها را جداگانه بررسی کن:
    - چه عبارت یا اطلاعاتی این مقدار را توجیه می‌کند؟
    - آیا مقدار واقعاً از درخواست یا زمینه معتبر استخراج شده است؟
    - آیا می‌توان بدون این آرگومان درخواست را با قرارداد ابزار اجرا کرد؟
    اگر مقدار مستند نیست، یا پارامتر اختیاری و غیرضروری است،
    آن را حذف کن.

11. بررسی نهایی باید روی خود JSON انجام شود، نه فقط روی reason.
    نوشتن اینکه پارامتری مشخص نشده، مجوز ارسال مقدار حدسی آن نیست.
    بین توضیح تصمیم و آرگومان‌های خروجی هیچ تناقضی نباید وجود داشته باشد.

12. اگر پارامتر اجباری است و مقدار معتبر آن در دسترس نیست، ابزار را
    با مقدار حدسی فراخوانی نکن. مطابق سیاست مسیریابی، درخواست را
    برای دریافت اطلاعات لازم به مسیر chat هدایت کن.

13. آرگومان‌های خروجی باید فقط شامل پارامترهای تعریف‌شده در
    input_schema همان ابزار باشند. هیچ پارامتر اضافه‌ای تولید نکن.

    
قوانین RAG:
- درخواست‌های مربوط به محتوای اسناد، راهنمای کاربری ERP، فرم‌های سازمانی،
  فیلدهای فرم‌ها، کاربرد اسناد و مراحل انجام فرایندهای ثبت‌شده در سیستم
  باید برای جست‌وجو در اسناد بررسی شوند.
- تعریفی یا آموزشی بودن سؤال، به‌تنهایی دلیل انتخاب مسیر chat نیست.
- اگر کاربر درباره ماهیت، کاربرد، نحوه تکمیل یا مراحل ثبت یک فرم یا سند
  سازمانی سؤال می‌کند، در صورت مرتبط بودن با اسناد ERP، مسیر rag را انتخاب کن.
- اگر درخواست هم به اطلاعات عملیاتی زنده از ابزارهای MCP و هم به محتوای
  اسناد نیاز دارد، مسیر tool_and_rag را انتخاب کن.
- اگر درخواست یک گفت‌وگوی عادی، احوال‌پرسی یا سؤال عمومی مستقل از اسناد
  و ابزارهاست، مسیر chat را انتخاب کن.
- اگر جست‌وجوی اسناد لازم است، rag_query را به‌صورت یک عبارت روشن و
  مناسب برای جست‌وجو بنویس؛ ترجیحاً اصطلاح اصلی کاربر را حفظ کن.
- اگر ابزار لازم نیست، tool_calls باید آرایه خالی باشد.


قوانین خروجی:
- confidence باید عددی بین 0 و 1 باشد.
- reason باید دلیل کوتاه تصمیم را توضیح دهد.
- خروجی باید فقط یک JSON معتبر باشد؛ بدون Markdown یا متن اضافی.
- ساختار خروجی دقیقاً به این صورت باشد:

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
    """Validate required fields and JSON Schema types."""
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
        "null": lambda value: value is None,
    }

    def matches_type(value: Any, expected_type: Any) -> bool:
        """Support both a single JSON Schema type and a type list."""
        expected_types = (
            expected_type
            if isinstance(expected_type, list)
            else [expected_type]
        )

        # Unknown or omitted type: no type assertion can be made here.
        if not expected_types or expected_types == [None]:
            return True

        return any(
            type_checks.get(type_name, lambda _: False)(value)
            for type_name in expected_types
            if isinstance(type_name, str)
        )

    for argument_name, value in arguments.items():
        property_schema = properties[argument_name]

        if not isinstance(property_schema, dict):
            continue

        expected_type = property_schema.get("type")

        if not matches_type(value, expected_type):
            raise ValueError(
                f"Argument '{argument_name}' for tool '{tool['name']}' "
                f"must have type '{expected_type}'."
            )

        expected_types = (
            expected_type
            if isinstance(expected_type, list)
            else [expected_type]
        )

        if "array" not in expected_types or not isinstance(value, list):
            continue

        item_schema = property_schema.get("items", {})

        if not isinstance(item_schema, dict):
            continue

        item_type = item_schema.get("type")

        if item_type is None:
            continue

        for index, item in enumerate(value):
            if not matches_type(item, item_type):
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

        schema = tools_by_name[name]["input_schema"]
        required = set(schema.get("required", []))

        # Remove null values only from optional arguments.
        # Required arguments remain so that validation rejects null.
        arguments = {
            key: value
            for key, value in arguments.items()
            if value is not None or key in required
        }

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
