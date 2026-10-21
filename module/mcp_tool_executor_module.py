
import asyncio
import json
from typing import Any

from mcp import ClientSession

from module.decision_module import decide_route
from module.mcp_module import close_mcp, connect_mcp
from module.tool_module import execute_tool, get_available_tools


def _tool_value(tool: Any, key: str, default: Any = None) -> Any:
    """Read metadata from either a dictionary or an MCP tool object."""

    if isinstance(tool, dict):
        return tool.get(key, default)

    return getattr(tool, key, default)


def _tool_schema(tool: Any) -> dict[str, Any]:
    """Return the input schema in a consistent dictionary format."""

    schema = _tool_value(tool, "input_schema")
    if schema is None:
        schema = _tool_value(tool, "inputSchema")

    if schema is None:
        return {}

    if isinstance(schema, dict):
        return schema

    if hasattr(schema, "model_dump"):
        value = schema.model_dump(mode="json")
        return value if isinstance(value, dict) else {}

    return {}


def _build_tool_catalog(
    tools: list[Any],
) -> dict[str, Any]:
    """Index the current MCP catalog without hardcoded tool names."""

    catalog: dict[str, Any] = {}

    for tool in tools:
        name = _tool_value(tool, "name")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "MCP returned a tool without a valid name."
            )

        if name in catalog:
            raise ValueError(
                f"MCP returned a duplicate tool name: {name}"
            )

        catalog[name] = tool

    return catalog


def _validate_json_type(
    name: str,
    value: Any,
    expected_type: str,
) -> None:
    """Validate common JSON Schema types without tool-specific rules."""

    valid = True

    if expected_type == "string":
        valid = isinstance(value, str)

    elif expected_type == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)

    elif expected_type == "number":
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    elif expected_type == "boolean":
        valid = isinstance(value, bool)

    elif expected_type == "object":
        valid = isinstance(value, dict)

    elif expected_type == "array":
        valid = isinstance(value, list)

    elif expected_type == "null":
        valid = value is None

    # Unknown/custom schema types are not guessed here.
    if not valid:
        raise ValueError(
            f"Invalid type for argument '{name}': "
            f"expected {expected_type}, "
            f"received {type(value).__name__}."
        )


def _validate_tool_call(
    tool_name: str,
    arguments: Any,
    available_tools: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Validate a model-selected tool call against live MCP metadata."""

    if tool_name not in available_tools:
        raise ValueError(
            f"Selected tool is not available on MCP server: {tool_name}"
        )

    if not isinstance(arguments, dict):
        raise ValueError(
            f"Arguments for '{tool_name}' must be a dictionary."
        )

    tool = available_tools[tool_name]
    schema = _tool_schema(tool)

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    if not isinstance(properties, dict):
        raise ValueError(
            f"Invalid input schema for MCP tool '{tool_name}'."
        )

    if not isinstance(required, list):
        raise ValueError(
            f"Invalid required-arguments schema for '{tool_name}'."
        )

    unknown = set(arguments) - set(properties)

    if unknown:
        raise ValueError(
            f"Unknown arguments for '{tool_name}': {sorted(unknown)}"
        )

    missing = [
        name for name in required
        if name not in arguments
    ]

    if missing:
        raise ValueError(
            f"Missing required arguments for '{tool_name}': {missing}"
        )

    for name, value in arguments.items():
        property_schema = properties[name]

        if not isinstance(property_schema, dict):
            continue

        expected_type = property_schema.get("type")

        if isinstance(expected_type, str):
            _validate_json_type(name, value, expected_type)

        if (
            isinstance(value, list)
            and isinstance(property_schema.get("items"), dict)
        ):
            item_type = property_schema["items"].get("type")

            if isinstance(item_type, str):
                for item in value:
                    _validate_json_type(
                        f"{name}[]",
                        item,
                        item_type,
                    )

    return tool_name, arguments


def _format_tool_result(result: Any) -> str:
    """Convert MCP response content to readable text."""

    content = _tool_value(result, "content")

    if not content:
        return str(result)

    parts: list[str] = []

    for item in content:
        item_text = _tool_value(item, "text")

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


async def _route_and_execute_question(
    question: str,
) -> dict[str, Any]:
    """Discover tools, decide the route, and execute selected tool calls."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    http_client = None
    stream_context = None
    session: ClientSession | None = None

    try:
        # 1. Read the live tool catalog.
        http_client, stream_context, session = await connect_mcp()

        tools = await get_available_tools(session=session)
        available_tools = _build_tool_catalog(tools)

        # 2. Let the decision model select a route and tool calls.
        decision = decide_route(
            query=question,
            available_tools=tools,
        )

        route = decision["route"]

        response: dict[str, Any] = {
            "route": route,
            "rag_query": decision.get("rag_query"),
            "confidence": decision.get("confidence"),
            "reason": decision.get("reason"),
            "tool_results": [],
        }

        # 3. Chat and RAG-only routes require no tool execution.
        if route in {"chat", "rag"}:
            return response

        # 4. Execute every validated tool call.
        tool_calls = decision.get("tool_calls", [])

        if not tool_calls:
            raise ValueError(
                f"Route '{route}' requires at least one tool call."
            )

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            arguments = tool_call["arguments"]

            tool_name, arguments = _validate_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                available_tools=available_tools,
            )

            result = await execute_tool(
                session=session,
                tool_name=tool_name,
                arguments=arguments,
            )

            is_error = bool(_tool_value(result, "isError", False))

            # Preserve MCP error details instead of replacing them with
            # a generic exception. This allows the caller to handle
            # permission-denied and other server errors appropriately.
            response["tool_results"].append({
                "tool_name": tool_name,
                "arguments": arguments,
                "result": _format_tool_result(result),
                "is_error": is_error,
            })

        return response

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


def route_and_execute_question(
    question: str,
) -> dict[str, Any]:
    """Synchronous API returning the route and all tool results."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _route_and_execute_question(question)
        )

    raise RuntimeError(
        "route_and_execute_question cannot run inside an active "
        "event loop. Use the async implementation directly."
    )


def execute_mcp_for_question(
    question: str,
) -> str | None:
    """
    Backward-compatible adapter for the current conversation module.

    Returns formatted tool results when tools were selected.
    Returns None for chat and RAG-only routes.
    """

    response = route_and_execute_question(question)
    tool_results = response["tool_results"]

    if not tool_results:
        return None

    sections: list[str] = []

    for item in tool_results:
        status = (
            "MCP ERROR"
            if item["is_error"]
            else "SUCCESS"
        )

        sections.append(
            f"Tool: {item['tool_name']}\n"
            f"Arguments: "
            f"{json.dumps(item['arguments'], ensure_ascii=False)}\n"
            f"Status: {status}\n"
            f"Result:\n{item['result']}"
        )

    return "\n\n".join(sections)
