from datetime import datetime, timedelta

import module.chat_memory_module as chat_memory
from module.message_model import Message


def make_message(user_id, chat_id, role, content, created_at=None):
    return Message(
        user_id=user_id,
        chat_id=chat_id,
        role=role,
        content=content,
        created_at=created_at or datetime.now(),
    )


def test_create_and_save_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(
        chat_memory,
        "MEMORY_FILE_PATH",
        str(tmp_path / "chat_memory.json"),
    )

    chat = chat_memory.create_chat(
        user_id="user_001",
        title="گفتگوی اول",
    )
    chat_memory.save_chat(chat)

    chats = chat_memory.get_user_chats("user_001")

    assert len(chats) == 1
    assert chats[0].chat_id == chat.chat_id
    assert chats[0].title == "گفتگوی اول"


def test_messages_are_isolated_by_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(
        chat_memory,
        "MEMORY_FILE_PATH",
        str(tmp_path / "chat_memory.json"),
    )

    chat_1 = chat_memory.create_chat("user_001", "گفتگوی اول")
    chat_2 = chat_memory.create_chat("user_001", "گفتگوی دوم")

    chat_memory.save_chat(chat_1)
    chat_memory.save_chat(chat_2)

    chat_memory.save_message(
        make_message(
            "user_001",
            chat_1.chat_id,
            "user",
            "پیام گفتگوی اول",
        )
    )
    chat_memory.save_message(
        make_message(
            "user_001",
            chat_2.chat_id,
            "user",
            "پیام گفتگوی دوم",
        )
    )

    messages = chat_memory.get_chat_messages(
        "user_001",
        chat_1.chat_id,
    )

    assert len(messages) == 1
    assert messages[0].content == "پیام گفتگوی اول"


def test_users_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(
        chat_memory,
        "MEMORY_FILE_PATH",
        str(tmp_path / "chat_memory.json"),
    )

    chat_1 = chat_memory.create_chat("user_001", "گفتگوی کاربر اول")
    chat_2 = chat_memory.create_chat("user_002", "گفتگوی کاربر دوم")

    chat_memory.save_chat(chat_1)
    chat_memory.save_chat(chat_2)

    chat_memory.save_message(
        make_message(
            "user_001",
            chat_1.chat_id,
            "user",
            "اطلاعات کاربر اول",
        )
    )
    chat_memory.save_message(
        make_message(
            "user_002",
            chat_2.chat_id,
            "user",
            "اطلاعات کاربر دوم",
        )
    )

    user_1_chats = chat_memory.get_user_chats("user_001")
    user_2_chats = chat_memory.get_user_chats("user_002")

    assert [c.chat_id for c in user_1_chats] == [chat_1.chat_id]
    assert [c.chat_id for c in user_2_chats] == [chat_2.chat_id]

    user_1_messages = chat_memory.get_chat_messages(
        "user_001",
        chat_1.chat_id,
    )
    assert all(m.user_id == "user_001" for m in user_1_messages)


def test_recent_messages_returns_latest_in_chronological_order(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        chat_memory,
        "MEMORY_FILE_PATH",
        str(tmp_path / "chat_memory.json"),
    )

    chat = chat_memory.create_chat("user_001", "گفتگو")
    chat_memory.save_chat(chat)

    base_time = datetime(2026, 1, 1, 10, 0, 0)

    for index in range(4):
        chat_memory.save_message(
            make_message(
                "user_001",
                chat.chat_id,
                "user",
                f"پیام {index}",
                base_time + timedelta(seconds=index),
            )
        )

    messages = chat_memory.get_recent_messages(
        user_id="user_001",
        chat_id=chat.chat_id,
        limit=2,
    )

    assert [m.content for m in messages] == ["پیام 2", "پیام 3"]


def test_delete_chat_deletes_only_its_messages(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        chat_memory,
        "MEMORY_FILE_PATH",
        str(tmp_path / "chat_memory.json"),
    )

    chat_1 = chat_memory.create_chat("user_001", "گفتگوی اول")
    chat_2 = chat_memory.create_chat("user_001", "گفتگوی دوم")
    other_user_chat = chat_memory.create_chat(
        "user_002",
        "گفتگوی کاربر دیگر",
    )

    for chat in (chat_1, chat_2, other_user_chat):
        chat_memory.save_chat(chat)

    for user_id, chat in (
        ("user_001", chat_1),
        ("user_001", chat_2),
        ("user_002", other_user_chat),
    ):
        chat_memory.save_message(
            make_message(
                user_id,
                chat.chat_id,
                "user",
                f"پیام {chat.title}",
            )
        )

    # A user must not be able to delete another user's chat.
    assert chat_memory.delete_chat(
        user_id="user_001",
        chat_id=other_user_chat.chat_id,
    ) is False

    # Delete one chat and verify the other chats remain.
    assert chat_memory.delete_chat(
        user_id="user_001",
        chat_id=chat_1.chat_id,
    ) is True

    assert chat_memory.get_chat_messages(
        "user_001",
        chat_1.chat_id,
    ) == []

    remaining_user_1_chats = chat_memory.get_user_chats("user_001")
    assert [c.chat_id for c in remaining_user_1_chats] == [
        chat_2.chat_id
    ]

    remaining_user_2_chats = chat_memory.get_user_chats("user_002")
    assert [c.chat_id for c in remaining_user_2_chats] == [
        other_user_chat.chat_id
    ]
