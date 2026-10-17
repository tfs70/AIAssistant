import os
from datetime import datetime

import streamlit as st

from module.chat_memory_module import (
    create_chat,
    get_chat_messages,
    get_user_chats,
    save_chat,
    update_chat,
)
from module.conversation_model import ConversationRequest
from module.conversation_module import send_message
from module.generator_module import generate_answer
from module.mcp_tool_executor_module import execute_mcp_for_question


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

APP_TITLE = "Persis AI Assistant"

# فقط برای اجرای محلی و آزمایشی است؛ احراز هویت واقعی نیست.
USER_ID = os.getenv(
    "AIASSISTANT_USER_ID",
    "streamlit_local_user",
)

# در صورت آماده بودن ایندکس Chroma، این گزینه را فعال کن.
DEFAULT_USE_RAG = False


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Persis AI Assistant")
st.caption("دستیار هوشمند متصل به سیستم ERP")


# ---------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------

def load_user_chats():
    """دریافت گفتگوهای کاربر فعلی."""
    return get_user_chats(user_id=USER_ID)


def create_new_chat() -> str:
    """ایجاد و ذخیره یک گفتگوی جدید."""
    chat = create_chat(
        user_id=USER_ID,
        title="گفتگوی جدید",
    )

    save_chat(chat=chat)

    return chat.chat_id


def get_chat_title(chat_id: str) -> str:
    """دریافت عنوان گفتگو."""
    chats = load_user_chats()

    chat = next(
        (
            item
            for item in chats
            if item.chat_id == chat_id
        ),
        None,
    )

    return chat.title if chat else "گفتگوی جدید"


def update_chat_title(
    chat_id: str,
    message: str,
) -> None:
    """تنظیم عنوان گفتگو بر اساس اولین پیام کاربر."""

    chats = load_user_chats()

    chat = next(
        (
            item
            for item in chats
            if item.chat_id == chat_id
        ),
        None,
    )

    if chat is None:
        return

    if chat.title == "گفتگوی جدید":
        title = " ".join(message.strip().split())

        if len(title) > 45:
            title = title[:45] + "..."

        chat.title = title or "گفتگوی جدید"
        chat.updated_at = datetime.now()

        update_chat(chat=chat)


def render_chat_history(chat_id: str) -> None:
    """نمایش تاریخچه کامل گفتگوی انتخاب‌شده."""

    messages = get_chat_messages(
        user_id=USER_ID,
        chat_id=chat_id,
    )

    for message in messages:
        role = (
            message.role
            if message.role in ("user", "assistant")
            else "assistant"
        )

        with st.chat_message(role):
            st.markdown(message.content)


# ---------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------

if "selected_chat_id" not in st.session_state:
    chats = load_user_chats()

    if chats:
        st.session_state.selected_chat_id = chats[0].chat_id
    else:
        st.session_state.selected_chat_id = create_new_chat()


if "use_rag" not in st.session_state:
    st.session_state.use_rag = DEFAULT_USE_RAG


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("گفتگوها")

    if st.button(
        "➕ گفتگوی جدید",
        use_container_width=True,
    ):
        st.session_state.selected_chat_id = create_new_chat()
        st.rerun()

    st.divider()

    use_rag = st.checkbox(
        "استفاده از RAG",
        value=st.session_state.use_rag,
        help=(
            "برای پرسش‌هایی که به اسناد ایندکس‌شده نیاز دارند "
            "این گزینه را فعال کن."
        ),
    )

    st.session_state.use_rag = use_rag

    st.divider()

    chats = load_user_chats()

    if not chats:
        st.caption("هنوز گفتگویی وجود ندارد.")

    for chat in chats:
        is_selected = (
            chat.chat_id == st.session_state.selected_chat_id
        )

        button_label = (
            f"● {chat.title}"
            if is_selected
            else chat.title
        )

        if st.button(
            button_label,
            key=f"chat_{chat.chat_id}",
            use_container_width=True,
        ):
            st.session_state.selected_chat_id = chat.chat_id
            st.rerun()


# ---------------------------------------------------------
# Current chat
# ---------------------------------------------------------

chat_id = st.session_state.selected_chat_id

st.subheader(get_chat_title(chat_id))

render_chat_history(chat_id=chat_id)


# ---------------------------------------------------------
# Message input and processing
# ---------------------------------------------------------

user_message = st.chat_input(
    "پیام خود را بنویسید..."
)

if user_message and user_message.strip():
    question = user_message.strip()

    # پیام کاربر را فوراً در رابط نمایش می‌دهیم.
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner(
                "در حال بررسی سؤال و دریافت پاسخ..."
            ):
                request = ConversationRequest(
                    user_id=USER_ID,
                    chat_id=chat_id,
                    message=question,
                )

                response = send_message(
                    request=request,
                    generate_answer=generate_answer,
                    use_rag=st.session_state.use_rag,
                    tool_executor=execute_mcp_for_question,
                )

                st.markdown(response.message)

        # send_message ممکن است شناسه جدیدی برگرداند.
        st.session_state.selected_chat_id = response.chat_id

        update_chat_title(
            chat_id=response.chat_id,
            message=question,
        )

        # تاریخچه از فایل حافظه دوباره خوانده می‌شود.
        st.rerun()

    except Exception as exc:
        st.error(
            "در پردازش پیام خطایی رخ داد. "
            "وضعیت Ollama، اتصال MCP و لاگ ترمینال را بررسی کن."
        )

        # جزئیات فقط در کنسول محلی ثبت می‌شود.
        print(
            f"[Streamlit Chat Error] "
            f"chat_id={chat_id}, error={exc!r}"
        )