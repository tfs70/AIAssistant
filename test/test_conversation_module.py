
from unittest.mock import Mock

import pytest

from module.conversation_model import ConversationRequest
from module import conversation_module as conversation


@pytest.fixture
def setup_conversation(monkeypatch):
    monkeypatch.setattr(
        conversation,
        "_may_contain_personal_fact",
        lambda message: False,
    )
    monkeypatch.setattr(
        conversation,
        "get_recent_messages",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        conversation,
        "get_user_facts",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        conversation,
        "build_context",
        lambda **kwargs: "BASE CONTEXT",
    )
    monkeypatch.setattr(
        conversation,
        "create_chat",
        lambda **kwargs: Mock(chat_id="chat-1"),
    )
    monkeypatch.setattr(
        conversation,
        "save_chat",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        conversation,
        "save_message",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        conversation,
        "get_user_chats",
        lambda **kwargs: [],
    )

    sync_mock = Mock(
        return_value={
            "added": 0,
            "updated": 0,
            "skipped": 0,
            "deleted": 0,
        }
    )
    retrieve_mock = Mock(return_value="RAG DOCUMENT CONTENT")

    monkeypatch.setattr(
        conversation,
        "sync_rag_documents",
        sync_mock,
    )
    monkeypatch.setattr(
        conversation,
        "retrieve_context",
        retrieve_mock,
    )

    return sync_mock, retrieve_mock


def make_request(message="سلام", chat_id="chat-1"):
    return ConversationRequest(
        user_id="test-user",
        chat_id=chat_id,
        message=message,
    )


@pytest.mark.parametrize(
    ("route", "use_rag", "expected_rag_calls"),
    [
        ("chat", True, 0),
        ("tool", True, 0),
        ("rag", True, 1),
        ("tool_and_rag", True, 1),
        ("rag", False, 0),
        ("tool_and_rag", False, 0),
    ],
)
def test_rag_runs_only_when_route_requires_it_and_enabled(
    setup_conversation,
    route,
    use_rag,
    expected_rag_calls,
):
    sync_mock, retrieve_mock = setup_conversation

    executor = Mock(
        return_value={
            "route": route,
            "rag_query": "document query",
            "tool_results": [],
        }
    )

    generator = Mock(return_value="پاسخ آزمایشی")

    conversation.send_message(
        request=make_request(),
        generate_answer=generator,
        use_rag=use_rag,
        tool_executor=executor,
    )

    assert retrieve_mock.call_count == expected_rag_calls
    assert sync_mock.call_count == expected_rag_calls


def test_tool_results_are_added_to_generation_context(setup_conversation):
    executor = Mock(
        return_value={
            "route": "tool",
            "rag_query": None,
            "tool_results": [
                {
                    "tool_name": "get_top_purchased_products",
                    "is_error": False,
                    "result": '[{"productId": 21, "purchasedCount": 1465}]',
                }
            ],
        }
    )

    generator = Mock(return_value="پاسخ بر اساس داده ابزار")

    conversation.send_message(
        request=make_request("کدام کالا بیشترین خرید را داشته است؟"),
        generate_answer=generator,
        use_rag=True,
        tool_executor=executor,
    )

    generated_context = generator.call_args.args[0]

    assert "get_top_purchased_products" in generated_context
    assert "1465" in generated_context
    assert "TRUSTED ERP TOOL RESULTS" in generated_context


def test_invalid_route_is_rejected(setup_conversation):
    executor = Mock(
        return_value={
            "route": "unknown",
            "rag_query": None,
            "tool_results": [],
        }
    )

    with pytest.raises(ValueError, match="Invalid route"):
        conversation.send_message(
            request=make_request(),
            generate_answer=Mock(return_value="پاسخ"),
            tool_executor=executor,
        )


def test_legacy_string_executor_is_rejected(setup_conversation):
    executor = Mock(return_value="متن نتیجه ابزار")

    with pytest.raises(TypeError, match="dictionary"):
        conversation.send_message(
            request=make_request(),
            generate_answer=Mock(return_value="پاسخ"),
            tool_executor=executor,
        )
