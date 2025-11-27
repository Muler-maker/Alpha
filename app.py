import os
import io
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

pd.options.display.float_format = "{:,.0f}".format

from charts import render_chart_from_answer, strip_chart_blocks
from data_loader import (
    load_orders,
    load_projection,
    load_metadata,
    preprocess_orders,
)
from consolidated import build_consolidated_df
from query_engine import answer_question_from_df


UI_TEXT = {
    # Data loading
    "loading_data": "Loading latest data from Google Sheets…",

    # Auth
    "password_label": "Password",
    "password_placeholder": "Enter password",
    "password_error": "Incorrect password.",

    # Chat input
    "chat_label": "Ask a question",
    "chat_placeholder": "What would you like Alpha to explore?",

    # Downloads
    "download_csv": "💾 Download chat (CSV)",
    "download_xlsx": "📊 Download chat (Excel)",
}

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
    /* Main content width + top padding */
    .block-container {
        padding-top: 0.5rem !important;
        max-width: 1100px;
    }

    :root {
        --isotopia-primary: #4A2E88;
        --isotopia-light:   #9C8AD0;
    }

    .data-badge {
        background-color: #E8F7E4;
        color: #267c3b;
        padding: 3px 7px;
        border-radius: 6px;
        font-size: 10px;
        display: inline-block;
        margin-top: -22px;
        margin-bottom: 2px;
    }

    /* ==========================
       Form-based chat input styling
       ========================== */

    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    [data-testid="stForm"] > div {
        background: transparent !important;
        border: none !important;
    }

    /* Hide the "Press Enter to submit" message */
    [data-testid="stForm"] [data-testid="InputInstructions"],
    [data-testid="stForm"] .instructions,
    [data-testid="stForm"] small,
    [data-testid="stForm"] [class*="instructions"],
    [data-testid="stForm"] [class*="InputInstructions"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        opacity: 0 !important;
    }

    /* Purple pill wrapper */
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
        background-color: #F6F0FF !important;
        border-radius: 999px !important;
        border: 1px solid #E0D0FF !important;
        display: flex !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        justify-content: space-between !important;
        padding: 4px 4px 4px 16px !important;
        gap: 8px !important;
        position: relative !important;
    }

    /* Input column (FIRST stColumn) */
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child {
        flex: 1 1 auto !important;
        background: transparent !important;
        min-width: 0 !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Button column (LAST stColumn) */
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child {
        flex: 0 0 auto !important;
        width: 38px !important;
        min-width: 38px !important;
        background: transparent !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
    }

    /* Target the text input wrapper */
    [data-testid="stForm"] [data-testid="stTextInput"],
    [data-testid="stForm"] [data-testid="stTextInput"] > div,
    [data-testid="stForm"] [data-testid="stTextInput"] > div > div {
        background: transparent !important;
        border: none !important;
    }

    [data-testid="stForm"] [data-testid="stTextInput"] input {
        background: transparent !important;
        border: none !important;
        padding: 8px 0px !important;
        font-size: 15px !important;
        color: #2D1B56 !important;
        box-shadow: none !important;
        width: 100% !important;
    }

    [data-testid="stForm"] [data-testid="stTextInput"] input::placeholder {
        color: #9CA3AF !important;
        opacity: 0.7 !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 2.2rem !important;
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        [data-testid="stForm"] [data-testid="stTextInput"] input {
            font-size: 14px !important;
            padding-top: 10px !important;
            padding-bottom: 10px !important;
        }

        .alpha-chat-footer-inner,
        .alpha-chat-input-wrapper {
            width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }

    [data-testid="stForm"] [data-testid="stTextInput"] input:focus {
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    [data-testid="stForm"] button[kind="primary"],
    [data-testid="stForm"] button[type="submit"],
    [data-testid="stForm"] button {
        background-color: var(--isotopia-light) !important;
        color: #6B7280 !important;
        border-radius: 999px !important;
        border: none !important;
        width: 38px !important;
        height: 38px !important;
        min-width: 38px !important;
        min-height: 38px !important;
        padding: 0 !important;
        box-shadow: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
    }

    [data-testid="stForm"] button[kind="primary"]:hover,
    [data-testid="stForm"] button[type="submit"]:hover,
    [data-testid="stForm"] button:hover {
        background-color: #6A5CA8 !important;
    }

    [data-testid="stForm"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child > div {
        background: transparent !important;
    }

    [data-testid="stForm"] button *,
    [data-testid="stForm"] button > * {
        background-color: transparent !important;
        background: transparent !important;
    }

    [data-testid="stForm"] button p {
        margin: 0 !important;
        font-size: 16px !important;
        line-height: 1 !important;
        color: #FFFFFF !important;
        background: transparent !important;
    }

    /* Kill the grey outer bar of st.chat_input (kept for reference) */
    [data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        background-color: #F6F0FF !important;
        border-radius: 999px !important;
        border: 1px solid #E0D0FF !important;
        box-shadow: none !important;
        padding-left: 18px !important;
    }

    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #2D1B56 !important;
        font-size: 15px !important;
    }

    [data-testid="stChatInput"] button[kind="primary"] {
        background-color: var(--isotopia-light) !important;
        color: #FFFFFF !important;
        border-radius: 999px !important;
        border: none !important;
        width: 40px !important;
        height: 40px !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] button[kind="primary"]:hover {
        background-color: #6A5CA8 !important;
    }

    .chat-bottom-spacer {
        height: 80px;
    }

    .alpha-chat-footer-outer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 8px 0;
        background: linear-gradient(
            to top,
            rgba(255,255,255,0.95),
            rgba(255,255,255,0.9),
            rgba(255,255,255,0.0)
        );
        z-index: 999;
    }

    .alpha-chat-footer-inner {
        max-width: 1100px;
        margin: 0 auto;
    }

    .alpha-chat-input-wrapper {
        width: 100%;
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

    with st.spinner(UI_TEXT["loading_data"]):
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
                UI_TEXT["password_label"],
                type="password",
                placeholder=UI_TEXT["password_placeholder"],
                label_visibility="collapsed",
            )
            if pwd:
                if pwd == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error(UI_TEXT["password_error"])
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
                    UI_TEXT["download_csv"],
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
                    UI_TEXT["download_xlsx"],
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

                # If this message has a chart block, render the chart(s)
                if role == "assistant" and "raw_answer" in msg:
                    charts = render_chart_from_answer(msg["raw_answer"])
                    if charts:
                        for chart in charts:
                            st.altair_chart(chart, use_container_width=True)

        # Spacer so last message is above fixed footer
        st.markdown('<div class="chat-bottom-spacer"></div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # end alpha-content-wrapper

    # ========== FIXED CHAT FOOTER ==========
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
                UI_TEXT["chat_label"],
                value="",
                label_visibility="collapsed",
                placeholder=UI_TEXT["chat_placeholder"],
            )
        with col_btn:
            submitted = st.form_submit_button("➤")

    st.markdown("</div></div></div>", unsafe_allow_html=True)

    # ---------- PROCESS NEW QUESTION ----------
    if submitted and prompt.strip():
        user_text = prompt.strip()

        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_text})

        # Generate raw answer (with potential chart blocks)
        try:
            raw_answer = answer_question_from_df(
                user_text,
                df,
                history=st.session_state.messages,
            )
        except Exception as e:
            raw_answer = f"An error occurred: {e}"

        # ---------------------------------
        # ✨ OPTIONAL GPT REFINEMENT LAYER
        # ---------------------------------
        refined_answer = raw_answer

        if client is not None:
            try:
                # Remove chart blocks before sending to GPT
                answer_without_charts = strip_chart_blocks(raw_answer)

                prompt_refine = f"""
You are Alpha, a senior data analyst.

Refine the text below so that it:
- sounds natural and non-robotic
- preserves ALL numbers exactly
- keeps tables in markdown format
- does NOT add or infer new data
- keeps the structure clear and concise

Text:
---
{answer_without_charts}
---
"""

                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You refine analytical outputs for business users.",
                        },
                        {"role": "user", "content": prompt_refine},
                    ],
                    temperature=0.2,
                )

                refined_answer = response.choices[0].message.content.strip()

            except Exception:
                # If refinement fails, fall back to raw answer
                refined_answer = strip_chart_blocks(raw_answer)

        # For display we use the refined text (no chart blocks)
        cleaned = refined_answer

        # Store assistant message with BOTH:
        # - content (refined text for chat)
        # - raw_answer (original with chart blocks for rendering)
        st.session_state.messages.append({
            "role": "assistant",
            "content": cleaned,
            "raw_answer": raw_answer,
        })

        # Rerun so the updated chat (and charts) are rendered above
        st.rerun()


if __name__ == "__main__":
    main()
