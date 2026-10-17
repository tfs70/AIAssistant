
import json
import re
from typing import Any

from module.conversation_model import ConversationRequest
from module.conversation_module import send_message
from module.generator_module import generate_answer
from module.mcp_tool_executor_module import execute_mcp_for_question


QUESTION = "سه تأمین‌کننده با بیشترین مجموع روزهای تأخیر را بده"


def normalize_digits(text: str) -> str:
    """Convert Persian and Arabic digits to English digits."""

    translation = str.maketrans(
        "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
        "01234567890123456789",
    )

    return text.translate(translation)


def normalize_number_text(text: str) -> str:
    """Normalize digits and remove common number separators."""

    text = normalize_digits(text)
    return re.sub(r"[,٬،\s]", "", text)


def parse_tool_result(tool_result: str) -> list[dict[str, Any]]:
    """Extract the actual JSON data returned by the ERP tool."""

    marker = "Result:\n"

    if marker not in tool_result:
        raise AssertionError(
            "The MCP result does not contain the expected Result section."
        )

    json_text = tool_result.split(marker, maxsplit=1)[1].strip()
    data = json.loads(json_text)

    if not isinstance(data, list) or not data:
        raise AssertionError(
            "The ERP tool returned no supplier records."
        )

    return data


def test_real_conversation_with_mcp() -> None:
    captured_tool_results: list[str] = []

    def real_tool_executor(question: str) -> str | None:
        """Execute the real MCP flow and retain its result for verification."""

        result = execute_mcp_for_question(question)

        if result is not None:
            captured_tool_results.append(result)

        return result

    
    def debug_generator(context: str) -> str:
        print("\n===== ACTUAL GENERATOR INPUT =====")
        print(context)

        return generate_answer(context)

    request = ConversationRequest(
        user_id="mcp_integration_test",
        chat_id="",
        message=QUESTION,
    )

    print("\n===== USER QUESTION =====")
    print(QUESTION)

    # Run the complete conversation using the real Generator and MCP.
    response = send_message(
        request=request,
        generate_answer=debug_generator,
        use_rag=False,
        tool_executor=real_tool_executor,
    )

    print("\n===== CONVERSATION RESPONSE =====")
    print("Chat ID:", response.chat_id)
    print("Answer:", response.message)

    # Verify that an actual ERP tool was executed.
    assert captured_tool_results, (
        "No ERP tool result was received. "
        "Check the decision model and MCP tool execution."
    )

    # This question should require an ERP tool.
    assert len(captured_tool_results) == 1, (
        f"Expected one ERP tool execution, "
        f"got {len(captured_tool_results)}."
    )

    # Parse the real data returned by MCP.
    records = parse_tool_result(captured_tool_results[0])

    print("\n===== ACTUAL ERP DATA =====")
    print(json.dumps(records, ensure_ascii=False, indent=2))

    answer = response.message.strip()

    assert answer, "The Generator returned an empty answer."

    normalized_answer = normalize_number_text(answer)

    # Check that the answer includes each supplier name and its
    # actual totalDelayDays value from the returned ERP records.
    for record in records:
        vendor_name = record.get("vendorName")
        total_delay_days = record.get("totalDelayDays")

        assert isinstance(vendor_name, str) and vendor_name.strip(), (
            f"Invalid vendorName in ERP record: {record}"
        )

        assert total_delay_days is not None, (
            f"Missing totalDelayDays in ERP record: {record}"
        )

        assert vendor_name in answer, (
            f"Supplier name missing from final answer: {vendor_name!r}"
        )

        expected_number = normalize_number_text(
            str(total_delay_days)
        )

        assert expected_number in normalized_answer, (
            f"Actual totalDelayDays value {total_delay_days!r} "
            f"for supplier {vendor_name!r} was not found in the answer."
        )

    print("\nAll integration checks passed.")
    print("Real MCP execution: PASSED")
    print("Real Generator: PASSED")
    print("Supplier names: PASSED")
    print("ERP delay values: PASSED")


if __name__ == "__main__":
    test_real_conversation_with_mcp()
    print("\ntest_real_conversation_with_mcp: PASSED")
