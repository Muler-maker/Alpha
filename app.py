import os
import json
import io
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
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


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

    .alpha-chat-input-wrapper {
        margin: 2rem auto 2.5rem auto;
        max-width: 720px;
        width: 100%;
    }

    /* Remove outer form frame */
    .alpha-chat-input-wrapper [data-testid="stForm"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* CHAT INPUT: Rounded, very light purple */
    .alpha-chat-input-wrapper [data-baseweb="base-input"] {
        background-color: #F6F0FF !important;      /* very light purple */
        color: #2D1B56 !important;
        border: none !important;
        border-radius: 999px !important;           /* pill shape */
        box-shadow: none !important;
        padding-left: 18px !important;
    }

    /* Subtle focus ring */
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

    /* SEND BUTTON — purple circle */
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
</style>
""",
    unsafe_allow_html=True,
)

# ================================
# 🧩 LOAD DATA
# ================================
def init_state():
def load_data_if_needed():
    if st.session_state.data_loaded:
        return

    with st.spinner("Loading latest data from Google Sheets…"):
        raw_orders = load_orders()
        orders_df = preprocess_orders(raw_orders)
        proj_df   = load_projection()
        meta_df   = load_metadata()

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

    # ---------------- AUTH ----------------
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

    # If no password or authenticated:
    load_data_if_needed()
    df = st.session_state.consolidated_df

    # Layout: left exports, right chat
    left, right = st.columns([1.1, 4], gap="large")

    # ---------------- RIGHT SIDE: CHAT ----------------
    with right:
        st.markdown(
            f"<div class='data-badge'>✔ Data synced · {len(df):,} rows · {len(df.columns)} columns</div>",
            unsafe_allow_html=True,
        )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        st.markdown('<div class="alpha-chat-input-wrapper">', unsafe_allow_html=True)
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

    # ---------------- PROCESS QUESTIONS ----------------
    if submitted and prompt.strip():
        txt = prompt.strip()
        st.session_state.messages.append({"role": "user", "content": txt})

        with right:
            with st.chat_message("assistant"):
                try:
                    reply = answer_question_from_df(
                        txt, df, history=st.session_state.messages
                    )
                except Exception as e:
                    reply = f"Error: {e}"

                cleaned = strip_chart_blocks(reply)
                st.markdown(cleaned)

                try:
                    render_chart_from_answer(reply)
                except:
                    pass

        st.session_state.messages.append({"role": "assistant", "content": cleaned})

    # ---------------- LEFT SIDE: EXPORTS ----------------
    with left:
        if st.session_state.messages:
            st.caption("Current session")

            export_df = pd.DataFrame(st.session_state.messages)

            # CSV
            csv_bytes = export_df.to_csv(index=False).encode()
            st.download_button(
                "💾 Download chat (CSV)",
                data=csv_bytes,
                file_name="alpha_chat.csv",
                mime="text/csv",
            )

            # Excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                export_df.to_excel(writer, index=False)
            st.download_button(
                "📊 Download chat (Excel)",
                data=buf.getvalue(),
                file_name="alpha_chat.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
