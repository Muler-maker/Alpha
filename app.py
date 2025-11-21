import os
import io
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from charts import render_chart_from_answer, strip_chart_blocks
from data_loader import (
    load_orders,
    load_projection,
    load_metadata,
    preprocess_orders,
)
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
/* Main container width */
.block-container {
    padding-top: 0.5rem;
    max-width: 1100px;
}

/* Brand colors */
:root {
    --isotopia-primary: #4A2E88;
    --isotopia-light:   #9C8AD0;
    --chat-bg:          #F6F0FF;
}

/* Centered inner content (badge, downloads, messages) */
.alpha-content-wrapper {
    max-width: 900px;
    margin: 0 auto;
}

/* Data badge */
.data-badge {
    background-color: #E8F7E4;
    color: #267c3b;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 13px;
    display: inline-block;
    margin-top: 10px;
    margin-bottom: 12px;
}

/* Download buttons row */
.download-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 18px;
}

/* Let Streamlit buttons look a bit nicer */
.download-row .stButton > button {
    border-radius: 8px !important;
}

/* Extra bottom space so messages are not hidden behind fixed chat bar */
.chat-bottom-spacer {
    height: 120px;
}

/* ==========================
   CHAT FOOTER (fixed)
   ========================== */

.alpha-chat-footer-outer {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 0.4rem 0.5rem 0.8rem 0.5rem;
    background: rgba(255, 255, 255, 0.97);
    z-index: 1000;
    box-shadow: 0 -2px 6px rgba(0, 0, 0, 0.05);
}

.alpha-chat-footer-inner {
    max-width: 900px;
    margin: 0 auto;
}

/* Wrapper for the form itself */
.alpha-chat-input-wrapper {
    margin: 0 auto;
    max-width: 720px;   /* desktop */
    width: 100%;        /* mobile */
}

/* Remove default form frame */
.alpha-chat-input-wrapper [data-testid="stForm"] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Align input + button */
.alpha-chat-input-wrapper [data-testid="column"] {
    display: flex;
    align-items: center;
}

/* Pill background for the text input */
.alpha-chat-input-wrapper [data-baseweb="base-input"] {
    background-color: var(--chat-bg) !important;
    color: #F6F0FF !important;
    border: none !important;
    border-radius: 999px !important;
    padding-left: 18px !important;
    box-shadow: none !important;
}

/* Focus ring */
.alpha-chat-input-wrapper [data-baseweb="base-input"]:focus-within {
    box-shadow: 0 0 0 2px #D6C3FF !important;
    border: none !important;
}

/* Inner text styling */
.alpha-chat-input-wrapper input[type="text"] {
    background-color: transparent !important;
    color: #2D1B56 !important;
    font-size: 15px !important;
}

/* Force pill look on any st.text_input in this area */
.alpha-chat-input-wrapper [data-testid="stTextInput"] > div > div {
    background-color: var(--chat-bg) !important;
    border-radius: 999px !important;
    border: none !important;
    box-shadow: none !important;
}

/* Send button (round) */
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

/* Ensure app body can scroll behind the fixed footer */
html, body, [data-testid="stAppViewContainer"] {
    overflow-x: hidden;
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
        st.write("")
        st.write("")
        st.write("")

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

    # ========== MAIN CONTENT (centered) ==========
    with st.container():
        st.markdown('<div class="alpha-content-wrapper">', unsafe_allow_html=True)

        # Badge
        st.markdown(
            f"<div class='data-badge'>✔ Data synced · "
            f"{len(df):,} rows · {len(df.columns)} columns</div>",
            unsafe_allow_html=True,
        )

        # Download buttons row
        if st.session_state.messages:
            st.markdown('<div class="download-row">', unsafe_allow_html=True)

            export_df = pd.DataFrame(st.session_state.messages)

            # CSV
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            col_csv, col_xlsx = st.columns(2)
            with col_csv:
                st.download_button(
                    "💾 Download chat (CSV)",
                    data=csv_bytes,
                    file_name="alpha_chat.csv",
                    mime="text/csv",
                    key="download_chat_csv",
                )

            # Excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                export_df.to_excel(writer, index=False, sheet_name="Chat")
            with col_xlsx:
                st.download_button(
                    "📊 Download chat (Excel)",
                    data=buf.getvalue(),
                    file_name="alpha_chat.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="download_chat_excel",
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # Chat history
        for msg in st.session_state.messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")

            if role == "user":
                avatar = "🧪"   # user icon
            else:
                avatar = "☢️"   # Alpha icon

            with st.chat_message(role, avatar=avatar):
                st.markdown(content)

        # Spacer so last message is above fixed footer
        st.markdown('<div class="chat-bottom-spacer"></div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # end alpha-content-wrapper

    # ========== FIXED CHAT FOOTER ==========
    # We render this once, outside the content wrapper, at the very end of the app.
    footer_html = """
<div class="alpha-chat-footer-outer">
  <div class="alpha-chat-footer-inner">
    <div class="alpha-chat-input-wrapper">
"""
    st.markdown(footer_html, unsafe_allow_html=True)

    with st.form("alpha-chat-form", clear_on_submit=True):
        col_input, col_btn = st.columns([12, 1])
        with col_input:
            prompt = st.text_input(
                "Ask a question",
                value="",
                label_visibility="collapsed",
                placeholder="Ask a question about orders, projections, or major events…",
            )
        with col_btn:
            submitted = st.form_submit_button("➤")

    st.markdown("</div></div></div>", unsafe_allow_html=True)

    # ---------- PROCESS NEW QUESTION ----------
    if submitted and prompt.strip():
        user_text = prompt.strip()

        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_text})

        # Generate answer
        try:
            raw_answer = answer_question_from_df(
                user_text,
                df,
                history=st.session_state.messages,
            )
        except Exception as e:
            raw_answer = f"An error occurred: {e}"

        cleaned = strip_chart_blocks(raw_answer)
        st.session_state.messages.append({"role": "assistant", "content": cleaned})

        # Force a rerun so the new messages appear above the fixed footer
        st.rerun()


if __name__ == "__main__":
    main()
