import os
import io
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from charts import render_chart_from_answer, strip_chart_blocks
from data_loader import load_orders, load_projection, load_metadata, preprocess_orders
from consolidated import build_consolidated_df
from query_engine import answer_question_from_df


# ================================
# 🔧 ENV + API
# ================================
load_dotenv()

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", ""))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

st.set_page_config(
    page_title="Alpha – Theranostics Engine",
    page_icon="🧬",
    layout="wide",
)

# ================================
# 🎨 GLOBAL CSS
# ================================
st.markdown(
    """
<style>
    /* Main page container */
    .block-container {
        padding-top: 0.5rem;
        max-width: 1100px;
    }

    :root {
        --isotopia-primary: #4A2E88;
        --isotopia-light:   #9C8AD0;
    }

    body, .main {
        background-color: #FFFFFF !important;
    }

    /* Center the data badge so it aligns with the chat */
    .data-badge-wrapper {
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .data-badge {
        background-color: #E8F7E4;
        color: #267c3b;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 13px;
        display: inline-block;
    }

    /* --------------------------
       CHAT INPUT (st.chat_input)
       -------------------------- */

    /* Make the area transparent, no grey strip */
    [data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        background-color: transparent !important;
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
        border-top: none !important;
        z-index: 999;
    }

    /* Center and narrow the inner container */
    [data-testid="stChatInput"] > div {
        max-width: 720px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Purple pill look for the input itself */
    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        background-color: #F6F0FF !important;
        color: #2D1B56 !important;
        border-radius: 999px !important;
        border: 1px solid #F0E8FF !important;
        box-shadow: none !important;
        padding-left: 16px !important;
        padding-right: 16px !important;
    }

    [data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
        box-shadow: 0 0 0 2px #D6C3FF !important;
        border: 1px solid #D6C3FF !important;
    }

    [data-testid="stChatInput"] input[type="text"] {
        background: transparent !important;
        color: #4A2E88 !important;
        font-size: 15px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ================================
# ⚙️ STATE
# ================================
def init_state():
    ss = st.session_state
    if "messages" not in ss:
        ss.messages = []
    if "data_loaded" not in ss:
        ss.data_loaded = False
    if "authenticated" not in ss:
        ss.authenticated = False


# ================================
# 📥 DATA LOADING
# ================================
def load_data_if_needed():
    ss = st.session_state
    if ss.data_loaded:
        return

    with st.spinner("Loading latest data from Google Sheets…"):
        raw_orders = load_orders()
        orders_df = preprocess_orders(raw_orders)
        proj_df = load_projection()
        meta_df = load_metadata()
        consolidated_df = build_consolidated_df(orders_df, proj_df, meta_df)

    ss.orders_df = orders_df
    ss.proj_df = proj_df
    ss.meta_df = meta_df
    ss.consolidated_df = consolidated_df
    ss.data_loaded = True


# ================================
# 🎨 HEADER
# ================================
def render_header():
    logo = Image.open("Isotopia.jpg")
    st.image(logo, use_container_width=True)


# ================================
# 🧠 MAIN APP
# ================================
def main():
    init_state()

    # Header always visible
    render_header()

    # ---------- AUTH ----------
    if APP_PASSWORD and not st.session_state.authenticated:
        st.write("")  # spacing
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            pwd = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password",
                label_visibility="collapsed",
            )
            if pwd:
                if pwd == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        return

    # ---------- DATA ----------
    load_data_if_needed()
    df = st.session_state.consolidated_df

    # Single centered container
    container = st.container()

    with container:
        # Data badge (centered)
        st.markdown(
            f"<div class='data-badge-wrapper'>"
            f"<span class='data-badge'>✔ Data synced · {len(df):,} rows · {len(df.columns)} columns</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Download buttons (only if there are messages)
        if st.session_state.messages:
            export_df = pd.DataFrame(st.session_state.messages)
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")

            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                export_df.to_excel(writer, index=False, sheet_name="Chat")

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "💾 Download chat (CSV)",
                    data=csv_bytes,
                    file_name="alpha_chat.csv",
                    mime="text/csv",
                    key="download_chat_csv",
                )
            with c2:
                st.download_button(
                    "📊 Download chat (Excel)",
                    data=excel_buf.getvalue(),
                    file_name="alpha_chat.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="download_chat_excel",
                )

        # Show existing conversation
        for msg in st.session_state.messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")

            avatar = "🧪" if role == "user" else "☢️"
            with st.chat_message(role, avatar=avatar):
                st.markdown(content)

        # --- Auto-scroll to bottom after rendering messages ---
        st.markdown(
            """
            <script>
            const main = window.parent.document.querySelector('section.main');
            if (main) { main.scrollTop = main.scrollHeight; }
            </script>
            """,
            unsafe_allow_html=True,
        )

        # Chat input at the bottom (sticky via CSS)
        prompt = st.chat_input(
            "Ask a question about orders, projections, or major events…"
        )

        # ---------- PROCESS NEW QUESTION ----------
        if prompt and prompt.strip():
            user_text = prompt.strip()

            # Save + show user message
            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.chat_message("user", avatar="🧪"):
                st.markdown(user_text)

            # Alpha answer
            with st.chat_message("assistant", avatar="☢️"):
                try:
                    with st.spinner("Thinking…"):
                        raw_answer = answer_question_from_df(
                            user_text,
                            df,
                            history=st.session_state.messages,
                        )
                except Exception as e:
                    raw_answer = f"An error occurred: {e}"

                cleaned = strip_chart_blocks(raw_answer)
                st.markdown(cleaned)

                # Try to render chart if present
                try:
                    render_chart_from_answer(raw_answer)
                except Exception:
                    pass

            # Store assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": cleaned}
            )


if __name__ == "__main__":
    main()
