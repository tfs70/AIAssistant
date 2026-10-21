import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import os
import traceback
from datetime import datetime

import streamlit as st

from module.chat_memory_module import (
    create_chat,
    delete_chat,
    get_chat_messages,
    get_user_chats,
    save_chat,
    update_chat,
)
from module.conversation_module import ConversationRequest, send_message
from module.generator_module import generate_answer
from module.mcp_tool_executor_module import route_and_execute_question


APP_TITLE = "Persis AI Assistant"
DEFAULT_USER_ID = os.getenv("AIASSISTANT_USER_ID", "User_1")
DEFAULT_USE_RAG = True


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💬",
    layout="wide",
)

st.title(APP_TITLE)
st.caption("دستیار هوشمند متصل به سیستم ERP")


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "user_id" not in st.session_state:
    st.session_state.user_id = DEFAULT_USER_ID

if "selected_chat_id" not in st.session_state:
    st.session_state.selected_chat_id = None

if "use_rag" not in st.session_state:
    st.session_state.use_rag = DEFAULT_USE_RAG

if "pending_delete_chat_id" not in st.session_state:
    st.session_state.pending_delete_chat_id = None


def format_datetime(value) -> str:
    if not value:
        return ""

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value

    return value.strftime("%Y-%m-%d %H:%M:%S")


def get_chat_by_id(chats, chat_id):
    if not chat_id:
        return None

    for chat in chats:
        if chat.chat_id == chat_id:
            return chat

    return None


def create_new_chat(user_id: str) -> str:
    chat = create_chat(
        user_id=user_id,
        title="گفتگوی جدید",
    )
    save_chat(chat=chat)
    return chat.chat_id


def update_chat_title(
    chat_id: str,
    user_id: str,
    first_message: str,
) -> None:
    chats = get_user_chats(user_id=user_id)
    chat = get_chat_by_id(chats, chat_id)

    if chat is None:
        return

    if chat.title not in ("گفتگوی جدید", "New Chat"):
        return

    title = " ".join(first_message.strip().split())

    if len(title) > 45:
        title = title[:45].rstrip() + "..."

    chat.title = title or "گفتگوی جدید"
    update_chat(chat=chat)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("تنظیمات گفتگو")

    # Local testing only; this selector is not authentication.
    test_users = list(
        dict.fromkeys(
            [
                DEFAULT_USER_ID,
                "QA_USER_A",
                "QA_USER_B",
                "QA_USER_C",
            ]
        )
    )

    # Protect against a stale session value that is not in the options.
    if st.session_state.user_id not in test_users:
        st.session_state.user_id = DEFAULT_USER_ID

    current_user_id = st.session_state.user_id

    test_user_id = st.selectbox(
        "کاربر آزمایشی",
        options=test_users,
        index=test_users.index(current_user_id),
        key="test_user_selector",
    )

    # Handle user switching in exactly one place.
    if test_user_id != st.session_state.user_id:
        st.session_state.user_id = test_user_id
        st.session_state.selected_chat_id = None
        st.session_state.pending_delete_chat_id = None
        st.rerun()

    user_id = st.session_state.user_id

    st.divider()

    st.session_state.use_rag = st.checkbox(
        "استفاده از اسناد (RAG)",
        value=st.session_state.use_rag,
        help="در صورت فعال بودن، امکان جست‌وجو در اسناد فراهم است.",
        key="use_rag_checkbox",
    )

    st.divider()

    if st.button(
        "＋ گفتگوی جدید",
        key="create_new_chat",
        use_container_width=True,
    ):
        new_chat_id = create_new_chat(user_id)

        st.session_state.selected_chat_id = new_chat_id
        st.session_state.pending_delete_chat_id = None
        st.rerun()

    st.subheader("گفتگوهای قبلی")

    # Always load chats for the active user only.
    chats = get_user_chats(user_id=user_id)
    chat_ids = [chat.chat_id for chat in chats]

    # Never keep a selection that does not belong to this user's chat list.
    if st.session_state.selected_chat_id not in chat_ids:
        st.session_state.selected_chat_id = None

    if (
        st.session_state.pending_delete_chat_id
        and st.session_state.pending_delete_chat_id not in chat_ids
    ):
        st.session_state.pending_delete_chat_id = None

    if not chats:
        st.caption("هنوز گفتگویی وجود ندارد.")
    else:
        # ChatGPT-like vertical list instead of a chat selectbox.
        for chat in chats:
            chat_title = (chat.title or "").strip() or "گفتگوی بدون عنوان"
            is_selected = (
                chat.chat_id == st.session_state.selected_chat_id
            )

            title_col, delete_col = st.columns(
                [0.82, 0.18],
                gap="small",
            )

            with title_col:
                if st.button(
                    f"{'▸ ' if is_selected else ''}{chat_title}",
                    key=f"select_chat_{chat.chat_id}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                    help=chat_title,
                ):
                    if chat.chat_id != st.session_state.selected_chat_id:
                        st.session_state.selected_chat_id = chat.chat_id
                        st.session_state.pending_delete_chat_id = None
                        st.rerun()

            with delete_col:
                if st.button(
                    "🗑️",
                    key=f"request_delete_{chat.chat_id}",
                    use_container_width=True,
                    help="حذف این گفتگو",
                ):
                    st.session_state.pending_delete_chat_id = chat.chat_id
                    st.rerun()

        active_sidebar_chat = get_chat_by_id(
            chats,
            st.session_state.selected_chat_id,
        )

        if active_sidebar_chat is not None:
            st.caption(
                "آخرین تغییر: "
                f"{format_datetime(active_sidebar_chat.updated_at)}"
            )

    # Delete confirmation
    pending_delete_id = st.session_state.pending_delete_chat_id

    if pending_delete_id:
        pending_chat = get_chat_by_id(chats, pending_delete_id)

        if pending_chat is not None:
            st.warning(
                f"گفتگوی «{pending_chat.title or 'گفتگوی بدون عنوان'}» "
                "حذف شود؟"
            )

            confirm_col, cancel_col = st.columns(2)

            with confirm_col:
                if st.button(
                    "بله، حذف شود",
                    key="confirm_delete_chat",
                    type="primary",
                    use_container_width=True,
                ):
                    deleted = delete_chat(
                        user_id=user_id,
                        chat_id=pending_delete_id,
                    )

                    st.session_state.pending_delete_chat_id = None

                    if st.session_state.selected_chat_id == pending_delete_id:
                        st.session_state.selected_chat_id = None

                    if deleted:
                        st.toast("گفتگو حذف شد.")
                    else:
                        st.warning(
                            "گفتگو پیدا نشد یا امکان حذف آن وجود ندارد."
                        )

                    st.rerun()

            with cancel_col:
                if st.button(
                    "انصراف",
                    key="cancel_delete_chat",
                    use_container_width=True,
                ):
                    st.session_state.pending_delete_chat_id = None
                    st.rerun()


# --------------------------------------------------
# Active chat
# --------------------------------------------------

user_id = st.session_state.user_id
selected_chat_id = st.session_state.selected_chat_id

if not selected_chat_id:
    st.info(
        "برای شروع، یک گفتگوی جدید بسازید یا یکی از گفتگوهای قبلی "
        "را از فهرست سمت راست یا چپ انتخاب کنید."
    )
    st.stop()

chats = get_user_chats(user_id=user_id)
active_chat = get_chat_by_id(chats, selected_chat_id)

if active_chat is None:
    st.session_state.selected_chat_id = None
    st.rerun()


# --------------------------------------------------
# Edit chat title
# --------------------------------------------------

with st.form("edit_chat_title_form"):
    edited_title = st.text_input(
        "عنوان گفتگو",
        value=active_chat.title or "گفتگوی جدید",
        max_chars=100,
    )
    save_title = st.form_submit_button("ذخیره عنوان")

if save_title:
    edited_title = " ".join(edited_title.split())

    if not edited_title:
        st.warning("عنوان گفتگو نمی‌تواند خالی باشد.")
    else:
        active_chat.title = edited_title
        active_chat.updated_at = datetime.now()
        update_chat(chat=active_chat)
        st.success("عنوان گفتگو ذخیره شد.")
        st.rerun()


# --------------------------------------------------
# Render conversation history
# --------------------------------------------------

messages = get_chat_messages(
    user_id=user_id,
    chat_id=selected_chat_id,
)

for message in messages:
    role = message.role

    if role not in ("user", "assistant"):
        role = "assistant"

    with st.chat_message(role):
        st.markdown(message.content)

        if message.created_at:
            st.caption(format_datetime(message.created_at))


# --------------------------------------------------
# Chat input and response generation
# --------------------------------------------------

user_message = st.chat_input("پیام خود را بنویسید...")

if user_message and user_message.strip():
    user_message = user_message.strip()

    # Snapshot the identity and chat for this request.
    request_user_id = st.session_state.user_id
    request_chat_id = st.session_state.selected_chat_id
    request_use_rag = st.session_state.use_rag

    with st.chat_message("user"):
        st.markdown(user_message)

    try:
        with st.chat_message("assistant"):
            with st.spinner("در حال پردازش پرسش..."):
                request = ConversationRequest(
                    chat_id=request_chat_id,
                    user_id=request_user_id,
                    message=user_message,
                )

                response = send_message(
                    request=request,
                    generate_answer=generate_answer,
                    use_rag=request_use_rag,
                    tool_executor=route_and_execute_question,
                )

            st.markdown(response.message)

        update_chat_title(
            chat_id=response.chat_id,
            user_id=request_user_id,
            first_message=user_message,
        )

        st.session_state.selected_chat_id = response.chat_id
        st.rerun()

    except Exception as exc:
        st.error(
            "پردازش پیام ناموفق بود. جزئیات خطا در ترمینال "
            "اجرای Streamlit ثبت شده است."
        )
        print(
            f"[Streamlit UI] {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
