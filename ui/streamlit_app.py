import os
from datetime import datetime
import traceback
import streamlit as st

from module.chat_memory_module import (
    create_chat,
    delete_chat,
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
DEFAULT_USER_ID = os.getenv(
    "AIASSISTANT_USER_ID",
    "User_1",
)



# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
)

if "active_user_id" not in st.session_state:
    st.session_state.active_user_id = DEFAULT_USER_ID

USER_ID = st.session_state.active_user_id

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

# با تغییر کاربر، چت انتخاب‌شده کاربر قبلی نباید باقی بماند.
if st.session_state.get("selected_chat_user_id") != USER_ID:
    st.session_state.selected_chat_user_id = USER_ID

    st.session_state.pop("selected_chat_id", None)
    st.session_state.pending_delete_chat_id = None

if "selected_chat_id" not in st.session_state:
    chats = load_user_chats()

    if chats:
        st.session_state.selected_chat_id = chats[0].chat_id
    else:
        st.session_state.selected_chat_id = create_new_chat()


if "pending_delete_chat_id" not in st.session_state:
    st.session_state.pending_delete_chat_id = None 


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("کاربر آزمایشی")

    with st.form("switch_user_form"):
        requested_user_id = st.text_input(
            "User ID",
            value=USER_ID,
            help=(
                "شناسه کاربر برای آزمایش جداسازی چت‌ها. "
                "این فیلد جایگزین احراز هویت واقعی نیست."
            ),
        ).strip()

        switch_user = st.form_submit_button(
            "تغییر کاربر",
            use_container_width=True,
        )

    if switch_user:
        if not requested_user_id:
            st.error("شناسه کاربر نمی‌تواند خالی باشد.")
        elif requested_user_id != USER_ID:
            st.session_state.active_user_id = requested_user_id
            st.rerun()

    st.divider()

    st.header("گفتگوها")

    if st.button(
        "➕ گفتگوی جدید",
        use_container_width=True,
    ):
        st.session_state.selected_chat_id = create_new_chat()
        st.rerun()

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

        chat_col, delete_col = st.columns(
            [0.84, 0.16],
            gap="small",
        )

        with chat_col:
            if st.button(
                button_label,
                key=f"chat_{chat.chat_id}",
                use_container_width=True,
            ):
                st.session_state.selected_chat_id = chat.chat_id
                st.session_state.pending_delete_chat_id = None
                st.rerun()

        with delete_col:
            if st.button(
                "🗑️",
                key=f"delete_chat_{chat.chat_id}",
                help="حذف این گفتگو",
                use_container_width=True,
            ):
                st.session_state.pending_delete_chat_id = chat.chat_id
                st.rerun()

        # تأیید حذف فقط برای گفتگوی انتخاب‌شده جهت حذف
        if (
            st.session_state.pending_delete_chat_id
            == chat.chat_id
        ):
            st.warning(f"گفتگوی «{chat.title}» حذف شود؟")

            confirm_col, cancel_col = st.columns(2)

            with confirm_col:
                if st.button(
                    "حذف گفتگو",
                    key=f"confirm_delete_{chat.chat_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    deleted = delete_chat(
                        user_id=USER_ID,
                        chat_id=chat.chat_id,
                    )

                    st.session_state.pending_delete_chat_id = None

                    if deleted:
                        if (
                            st.session_state.selected_chat_id
                            == chat.chat_id
                        ):
                            remaining_chats = load_user_chats()

                            if remaining_chats:
                                st.session_state.selected_chat_id = (
                                    remaining_chats[0].chat_id
                                )
                            else:
                                st.session_state.selected_chat_id = (
                                    create_new_chat()
                                )

                        st.rerun()
                    else:
                        st.error(
                            "گفتگو پیدا نشد یا اجازه حذف آن را نداری."
                        )

            with cancel_col:
                if st.button(
                    "انصراف",
                    key=f"cancel_delete_{chat.chat_id}",
                    use_container_width=True,
                ):
                    st.session_state.pending_delete_chat_id = None
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
                    use_rag=True,
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
            "جزئیات خطا در ترمینال ثبت شده است."
        )

        print(
            f"[Streamlit Chat Error] "
            f"chat_id={chat_id}, error={exc!r}"
        )

        traceback.print_exc()