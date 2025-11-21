import os
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
    /* ================================
       ALPHA CHAT INPUT — CLEAN RESET
    ================================== */

    /* Wrapper centered + wide */
    .alpha-chat-input-wrapper {
        margin: 2rem auto 2rem auto;
        max-width: 900px;         /* wider */
        width: 100%;
    }

    /* Remove ALL outer white frame around st.form */
    .alpha-chat-input-wrapper [data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* Remove the invisible border container Streamlit adds */
    .alpha-chat-input-wrapper [data-testid="stForm"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* Force text input pill shape and purple background */
    .alpha-chat-input-wrapper [data-testid="stTextInput"] > div > div {
        background-color: #F6F0FF !important;   /* light purple */
        border-radius: 999px !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Input text styling */
    .alpha-chat-input-wrapper input[type="text"] {
        background-color: transparent !important;
        color: #4A2E88 !important;              /* brand purple */
        font-size: 16px !important;
        padding-left: 14px !important;
    }

    /* Send button */
    .alpha-chat-input-wrapper .stFormSubmitButton > button {
        background-color: #9C8AD0 !important;
        color: white !important;
        border-radius: 50% !important;
        border: none !important;
        width: 42px !important;
        height: 42px !important;
        font-size: 20px !important;
        margin-left: 8px;
    }
    .alpha-chat-input-wrapper .stFormSubmitButton > button:hover {
        background-color: #7E6BB8 !important;
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
    # Center logo naturally within the main container
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

    # ---------- BADGE ----------
    st.markdown(
        f"<div class='alpha-badge-wrapper'>"
        f"<span class='data-badge'>✔ Data synced · {len(df):,} rows · {len(df.columns)} columns</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ---------- CHAT HISTORY ----------
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        avatar = "🧪" if role == "user" else "☢️"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

    # ---------- CHAT INPUT ----------
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

    # ---------- PROCESS QUESTION ----------
    if submitted and prompt.strip():
        txt = prompt.strip()

        # Store user message
        st.session_state.messages.append({"role": "user", "content": txt})

        avatar_user = "🧪"
        avatar_assistant = "☢️"

        # Echo user message immediately
        with st.chat_message("user", avatar=avatar_user):
            st.markdown(txt)

        # Assistant reply
        with st.chat_message("assistant", avatar=avatar_assistant):
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

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": cleaned})

    # ---------- DOWNLOADS ----------
    if st.session_state.messages:
        st.markdown("### Download current chat")
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
