
from module.conversation_model import ConversationRequest
from module.conversation_module import send_message
from module.mcp_tool_executor_module import execute_mcp_for_question


def fake_generator(context: str) -> str:
    print("\n===== FINAL GENERATOR CONTEXT =====")
    print(context)

    if "TRUSTED ERP TOOL RESULT" in context:
        return "نتیجه واقعی ابزار ERP دریافت شد و به مدل ارائه شد."

    return "برای این سؤال ابزار ERP لازم نیست."


def test_conversation_with_mcp():
    request = ConversationRequest(
        user_id="mcp_integration_test",
        chat_id="",
        message="سه تأمین‌کننده با بیشترین مجموع روزهای تأخیر را بده",
    )

    response = send_message(
        request=request,
        generate_answer=fake_generator,
        use_rag=False,
        tool_executor=execute_mcp_for_question,
    )

    print("\n===== CONVERSATION RESPONSE =====")
    print("Chat ID:", response.chat_id)
    print("Answer:", response.message)

    assert response.chat_id
    assert response.message
    assert "نتیجه واقعی ابزار ERP دریافت شد" in response.message


if __name__ == "__main__":
    test_conversation_with_mcp()
    print("\ntest_real_conversation_with_mcp: PASSED")
