import asyncio
import os

import httpx
from ollama import AsyncClient

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


# ============================================================
# Configuration
# ============================================================

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://localhost:6111/mcp"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "qwen3:1.7b"
)

JWT_TOKEN = os.getenv(
    "MCP_JWT_TOKEN",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzY29wZSI6InN0YWZmIiwiVXNlck5hbWUiOiJhZG1pbmlzdHJhdG9yIiwiTXVzdENoYW5nZVBhc3N3b3JkIjoiRmFsc2UiLCJSb2xlcyI6IlsxXSIsIkpzb25UZW5hbnRJZHMiOiJbMSwyXSIsIlVzZXJJZCI6IjEiLCJWZW5kb3JJZCI6IjAiLCJWZW5kb3JBY2NvdW50aW5nQ2VudGVySWQiOiIwIiwiQ3VzdG9tZXJJZCI6IjAiLCJDdXN0b21lckFjY291bnRpbmdDZW50ZXJJZCI6IjAiLCJrZXkiOiJQZXJzaXNVSSIsIm5iZiI6MTc5MDU5NTk0MCwiZXhwIjoxNzkwNjMxOTQwLCJpYXQiOjE3OTA1OTU5NDB9.Lqk_0iN4y8hmeEzTah4_8yias6VzeaH9ZB0KFt8G9So"
)


# ============================================================
# Ollama Client
# ============================================================

ollama = AsyncClient(
    host=OLLAMA_HOST
)


# ============================================================
# Convert MCP tools to Ollama format
# ============================================================

def mcp_tools_to_ollama_tools(mcp_tools):

    tools = []

    for tool in mcp_tools:

        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema
            }
        })

    return tools


# ============================================================
# MCP Result -> Text
# ============================================================

def mcp_result_to_text(result):

    parts = []

    for content in result.content:

        if hasattr(content, "text"):
            parts.append(content.text)
        else:
            parts.append(str(content))

    return "\n".join(parts)


# ============================================================
# Agent
# ============================================================

async def ask_agent(
    session: ClientSession,
    user_message: str,
    tools
):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an enterprise AI assistant. "
                "You have access to a set of tools for specific enterprise operations. "

                "Use a tool whenever the user's request requires real enterprise data "
                "or an external operation. "

                "Only answer enterprise-related questions when you have a suitable tool "
                "to retrieve the required information or perform the requested operation. "

                "If no suitable tool is available for the user's request, "
                "do not guess, invent, or fabricate an answer. "
                "Clearly state that you do not have the required expertise or tool "
                "to handle this request. "

                "Never claim to have access to enterprise data unless you actually "
                "retrieved it using an available tool."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    # --------------------------------------------------------
    # Agent Loop
    # --------------------------------------------------------

    while True:

        response = await ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=tools if tools else None
        )

        message = response["message"]

        # ----------------------------------------------------
        # No Tool Call
        # ----------------------------------------------------

        if not message.get("tool_calls"):

            return message.get(
                "content",
                ""
            )

        # ----------------------------------------------------
        # Add assistant response
        # ----------------------------------------------------

        messages.append(message)

        # ----------------------------------------------------
        # Execute Tool Calls
        # ----------------------------------------------------

        for tool_call in message["tool_calls"]:

            function = tool_call["function"]

            tool_name = function["name"]

            arguments = function.get(
                "arguments",
                {}
            )

            print()
            print(
                f"[Tool Call] {tool_name}"
            )

            print(
                f"[Arguments] {arguments}"
            )

            # -----------------------------------------------
            # Call MCP Server
            # -----------------------------------------------

            try:

                result = await session.call_tool(
                    tool_name,
                    arguments
                )

                result_text = mcp_result_to_text(
                    result
                )

            except Exception as e:

                result_text = (
                    f"Tool execution failed: {str(e)}"
                )

            print(
                f"[Tool Result] {result_text}"
            )

            # -----------------------------------------------
            # Send result back to Ollama
            # -----------------------------------------------

            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": result_text
            })


# ============================================================
# Main
# ============================================================

async def main():

    print()
    print("======================================")
    print(" Local Ollama + MCP Agent")
    print("======================================")

    print(
        f"Ollama     : {OLLAMA_HOST}"
    )

    print(
        f"Model      : {MODEL_NAME}"
    )

    print(
        f"MCP Server : {MCP_SERVER_URL}"
    )

    print(
        f"JWT        : {'Configured' if JWT_TOKEN else 'NOT CONFIGURED'}"
    )

    print()

    if not JWT_TOKEN:

        print(
            "WARNING: MCP_JWT_TOKEN is not configured."
        )

        print(
            "MCP requests will be sent without Authorization header."
        )

        print()

    # --------------------------------------------------------
    # HTTP Client for MCP
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # New MCP SDK versions do not accept:
    #
    #     streamable_http_client(..., headers={...})
    #
    # HTTP headers must be configured on the httpx client.
    #
    # --------------------------------------------------------

    http_headers = {}

    if JWT_TOKEN:

        http_headers["Authorization"] = (
            f"Bearer {JWT_TOKEN}"
        )

    async with httpx.AsyncClient(
        headers=http_headers
    ) as http_client:

        # ----------------------------------------------------
        # Connect to MCP Server
        # ----------------------------------------------------

        async with streamable_http_client(
            MCP_SERVER_URL,
            http_client=http_client
        ) as (
            read_stream,
            write_stream,
        ):

            async with ClientSession(
                read_stream,
                write_stream
            ) as session:

                # --------------------------------------------
                # Initialize MCP
                # --------------------------------------------

                await session.initialize()

                print(
                    "Connected to MCP Server."
                )

                # --------------------------------------------
                # Get Tools
                # --------------------------------------------

                tools_result = (
                    await session.list_tools()
                )

                mcp_tools = tools_result.tools

                print(
                    f"Available tools: {len(mcp_tools)}"
                )

                for tool in mcp_tools:

                    print(
                        f"  - {tool.name}"
                    )

                # --------------------------------------------
                # Convert MCP -> Ollama Tools
                # --------------------------------------------

                ollama_tools = (
                    mcp_tools_to_ollama_tools(
                        mcp_tools
                    )
                )

                print()
                print("Agent is ready.")
                print("Type 'exit' to quit.")
                print()

                # --------------------------------------------
                # Chat Loop
                # --------------------------------------------

                while True:

                    try:

                        user_input = input(
                            "You: "
                        ).strip()

                    except EOFError:

                        break

                    if not user_input:

                        continue

                    if user_input.lower() in (
                        "exit",
                        "quit"
                    ):

                        break

                    try:

                        answer = await ask_agent(
                            session,
                            user_input,
                            ollama_tools
                        )

                        print()
                        print(
                            f"Assistant: {answer}"
                        )
                        print()

                    except Exception as e:

                        print()
                        print(
                            f"ERROR: {e}"
                        )
                        print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())