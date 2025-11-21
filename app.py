import os
import io
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# Local modules
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
    .block-container {
        padding-top: 0.5rem;
        max-width: 1100px;
    }

    :root {
        --isotopia-primary: #4A2E88;
        --isotopia-light:   #9C8AD0;
    }

    .data-badge {
        background-color: #E8F7E4;
        color: #267c3b;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 13px;
        display: inline-block;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    /* ==========================
       CUSTOM CHAT FORM STYLING (FINAL)
       ========================== */

    /* Centered and shortened chat input */
    .alpha-chat-input-wrapper {
        margin: 2rem auto 2.5rem auto;
        max-width: 400px;          /* <<< NEW WIDTH */
        width: 100%;
    }

    /* Remove outer form frame */
    .alpha-chat-input-wrapper [data-testid="stForm"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* Chat input styling */
    .alpha-chat-input-wrapper [data-baseweb="base-input"] {
        background-color: #F6F0FF !important;
        color: #2D1B56 !important;
        border: none !important;
        border-radius: 999px !important;
        box-shadow: none !important;
        padding-left: 18px !important;
    }

    /* Focus ring */
    .alpha-chat-input-wrapper [data-baseweb="base-input"]:focus-within {
        box-shadow: 0 0 0 2px #D6C3FF !important;
        border: none !important;
    }

    /* Inner text */
    .alpha-chat-input-wrapper input[type="text"] {
        background: transparent !important;
        color: #2D1B56 !important;
        font-size: 15px !important;
    }

    /* Send button */
    .alpha-chat-input-wrapper .stFormSubmitButton > button {
        background-color: var(--isotopia-light) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 42px !important;
        height: 42px !important;
        font-size: 20px !important;
        padding: 0 !important;
        margin-left: 8px;
    }

    .alpha-chat-input-wrapper .stFormSubmitButton > button:hover {
        background-color: #6A5CA8 !important;
    }

    /* Align input + button */
    .alpha-chat-input-wrapper [data-testid="column"] {
        display: flex;
        align-items: center;
    }

    /* Enforce pill look on text input */
    [data-testid="stTextInput"] > div > div {
        background-color: #F6F0FF !important;
        border-radius: 999px !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stTextInput"] input[type="text"] {
        background-color: transparent !important;
        color: #4A2E88 !important;
        font-size: 15px !important;
    }

    /* Remove remaining outer frames */
    .alpha-chat-input-wrapper [data-testid="stForm"] > div {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        padding: 0 !important;
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
# 🧩 LOAD DATA
# ================================
def load_data_if_needed():
    if st.session_state.data_loaded:
        return

    with st.spinner("Loading latest data from Google Sheets…"):
        raw_orders = load_orders()
        orders_df = preprocess_orders(raw_orders)
        proj_df = load_projection()
        meta_df = load_metadata()

        consolidated = build_consolidated_df(orders_df, proj_df, meta_df)

    st.session_state.orders_df = orders_df
    st.session_state.proj_df = proj_df
    st.session_state.meta_df = meta_df
    st.session_state.consolidated_df = consolidated
    st.session_state.data_loaded = True


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

    # Always show logo
    render_header()

    # -------- AUTH --------
    if APP_PASSWORD and not st.session_state.authenticated:
        st.write("")
        st.write("")
        st.write("")

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
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

    # -------- DATA --------
    load_data_if_needed()
    df = st.session_state.consolidated_df

    # Layout: left exports, right chat
    left, right = st.columns([1.1, 4], gap="large")

    # ---------- RIGHT: CHAT ----------
    with right:
        st.markdown(
            f"<div class='data-badge'>✔ Data synced · "
            f"{len(df):,} rows · {len(df.columns)} columns</div>",
            unsafe_allow_html=True,
        )

        # show previous messages
        for msg in st.session_state.messages:
            role = msg.get("role", "assistant")
            with st.chat_message(role):
                st.markdown(msg["content"])

        # Chat input (narrow + centered)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        with st.form("alpha-chat-form", clear_on_submit=True):
            col_in, col_btn = st.columns([12, 1])

            with col_in:
                prompt = st.text_input(
                    "",
                    placeholder="Ask a question about orders, projections, or major events…",
                    label_visibility="collapsed",
                )

            with col_btn:
                submitted = st.form_submit_button("➤")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- PROCESS QUESTION ----------
    if "send_clicked" in locals() and send_clicked and prompt.strip():

        txt = prompt.strip()
        st.session_state.messages.append({"role": "user", "content": txt})

        with right:
            with st.chat_message("assistant"):
                try:
                    reply = answer_question_from_df(
                        txt,
                        df,
                        history=st.session_state.messages,
                    )
                except Exception as e:
                    reply = f"Error: {e}"

                cleaned = strip_chart_blocks(reply)
                st.markdown(cleaned)

                try:
                    render_chart_from_answer(reply)
                except Exception:
                    pass

        st.session_state.messages.append({"role": "assistant", "content": cleaned})

    # ---------- LEFT: EXPORTS ----------
    with left:
        if st.session_state.messages:
            st.caption("Current session")

            export_df = pd.DataFrame(st.session_state.messages)

            # CSV
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download chat (CSV)",
                data=csv_bytes,
                file_name="alpha_chat.csv",
                mime="text/csv",
            )

            # Excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                export_df.to_excel(writer, index=False, sheet_name="Chat")
            st.download_button(
                "📊 Download chat (Excel)",
                data=buf.getvalue(),
                file_name="alpha_chat.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )


if __name__ == "__main__":
    main()
