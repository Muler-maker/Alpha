import os
import json
import io
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
# 🔧 ENVIRONMENT + API
# ================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ================================
# 🎨 GLOBAL CSS
# ================================
st.markdown(
    """
<style>
    /* Main page container – pull everything up a bit and keep it narrow */
    .block-container {
        padding-top: 0.2rem;
        max-width: 1100px;
    }

    :root {
        --isotopia-primary: #4A2E88;
        --isotopia-light:   #9C8AD0;
        --chat-bg:          #F6F0FF;
    }

    .alpha-subheader {
        text-align: center;
        color: var(--isotopia-primary);
        font-size: 28px;
        font-weight: 700;
        margin-top: -40px;
    }

    .alpha-tagline {
        text-align: center;
        color: #666;
        font-size: 14px;
        margin-top: -10px;
        margin-bottom: 10px;
    }

    .sidebar-title {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: var(--isotopia-primary) !important;
        margin-bottom: 4px !important;
    }

    .data-badge {
        background-color: #E8F7E4;
        color: #267c3b;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 13px;
        display: inline-block;
        margin-top: 4px;
        margin-bottom: 16px;
    }

    /* Global buttons */
    .stButton > button {
        background-color: var(--isotopia-light) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.35rem 0.9rem !important;
        font-size: 14px !important;
    }
    .stButton > button:hover {
        background-color: #7E6BB8 !important;
        color: white !important;
    }

    /* ==========================
       CUSTOM CHAT FORM STYLING
       ========================== */

    /* Wrapper positioning */
    .alpha-chat-input-wrapper {
        margin: 1.5rem auto 2rem auto;
        max-width: 720px;
        width: 100%;
    }

    /* 1. REMOVE OUTER FRAME (The Form Border) */
    .alpha-chat-input-wrapper [data-testid="stForm"] {
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        background-color: transparent !important;
    }

    /* 2. COLOR THE INPUT BOX (#F6F0FF) */
    .alpha-chat-input-wrapper [data-baseweb="base-input"] {
        background-color: #F6F0FF !important;
        border: none !important;
        border-radius: 999px !important;
    }

    .alpha-chat-input-wrapper [data-testid="stTextInput"] {
        background-color: transparent !important;
    }
    .alpha-chat-input-wrapper input[type="text"] {
        background-color: transparent !important;
        color: #333 !important; 
    }
    
    .alpha-chat-input-wrapper [data-baseweb="base-input"]:focus-within {
        border: 1px solid #E4D8F8 !important;
        box-shadow: none !important;
    }

    /* 3. SUBMIT BUTTON STYLING */
    .alpha-chat-input-wrapper .stFormSubmitButton > button {
        background-color: #7E6BB8 !important;
        color: white !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        padding: 0 !important;
        font-size: 20px !important;
        line-height: 1 !important;
        border: none !important;
        margin-top: 0px !important;
    }
    .alpha-chat-input-wrapper .stFormSubmitButton > button:hover {
        background-color: #6A5CA8 !important;
    }

    .alpha-chat-input-wrapper [data-testid="column"] {
        display: flex;
        align-items: center;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ================================
# ⚙️ STATE
# ================================
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False


# ================================
# 📥 DATA LOADING
# ================================
def load_data_if_needed():
    if st.session_state.data_loaded:
        return

    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        with st.spinner("Loading latest data from Google Sheets..."):
            raw_orders = load_orders()
            orders_df = preprocess_orders(raw_orders)

            proj_df = load_projection()
            meta_df = load_metadata()

            consolidated_df = build_consolidated_df(orders_df, proj_df, meta_df)

    st.session_state.orders_df = orders_df
    st.session_state.proj_df = proj_df
    st.session_state.meta_df = meta_df
    st.session_state.consolidated_df = consolidated_df
    st.session_state.data_loaded = True


# ================================
# 🎨 HEADER
# ================================
def render_header():
    logo = Image.open("Isotopia.jpg")

    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        st.image(logo, use_container_width=True)


# ================================
# 🧠 MAIN APP
# ================================
def main():
    init_state()

    # -------- AUTH --------
    if APP_PASSWORD and not st.session_state.authenticated:
        col_l, col_c, col_r = st.columns([1, 3, 1])
        with col_c:
            st.write("Enter password")
            pwd = st.text_input(
                "Password",
                type="password",
                label_visibility="collapsed",
            )

            if pwd:
                if pwd == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.markdown(
                        "<p style='color:#DC2626; font-size:12px; "
                        "margin-top:6px;'>Incorrect password. Please try again.</p>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<p style='color:#9CA3AF; font-size:12px; margin-top:6px;'>"
                    "Access restricted – authorized Isotopia users only.</p>",
                    unsafe_allow_html=True,
                )
        return
    else:
        st.session_state.authenticated = True

    # -------- DATA --------
    load_data_if_needed()
    consolidated_df = st.session_state.get("consolidated_df")
    if consolidated_df is None:
        st.error("Could not load consolidated data.")
        return

    # Layout: left Chats panel + main content
    side_col, main_col = st.columns([1.1, 4], gap="large")

    # MAIN CONTENT
    with main_col:
        # Data badge
        st.markdown(
            f"<div class='data-badge'>✔ Data synced · "
            f"{len(consolidated_df):,} rows · {len(consolidated_df.columns)} columns"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Existing chat messages
        for msg in st.session_state.messages:
            role = msg.get("role", "assistant")
            avatar = "☢️" if role == "assistant" else "🧪"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])

        # --- Custom centered chat input bar ---
        st.markdown('<div class="alpha-chat-input-wrapper">', unsafe_allow_html=True)
        with st.form("alpha-chat-form", clear_on_submit=True):
            col_input, col_btn = st.columns([12, 1])
            with col_input:
                prompt = st.text_input(
                    "Ask a question",
                    value="",
                    key="alpha_chat_input",
                    label_visibility="collapsed",
                    placeholder=(
                        "Ask a question about orders, projections, or major events…"
                    ),
                )
            with col_btn:
                submitted = st.form_submit_button("➤")
        st.markdown("</div>", unsafe_allow_html=True)

    # -------- PROCESS NEW QUESTION (updates state before sidebar exports) --------
    if submitted and prompt.strip():
        user_input = prompt.strip()

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        with main_col:
            with st.chat_message("user", avatar="🧪"):
                st.markdown(user_input)

            # Assistant reply
            with st.chat_message("assistant", avatar="☢️"):
                try:
                    with st.spinner("Thinking…"):
                        reply = answer_question_from_df(
                            user_input,
                            consolidated_df,
                            history=st.session_state.messages,
                        )
                except Exception as e:
                    reply = f"An error occurred: {e}"

                cleaned = strip_chart_blocks(reply)
                st.markdown(cleaned)

                try:
                    render_chart_from_answer(reply)
                except Exception as e:
                    st.caption(f"⚠️ Chart error: {e}")

        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": cleaned})

    # LEFT PANEL (runs after messages are updated, so exports are up to date)
    with side_col:

        if st.session_state.messages:
            st.caption("Current session")

            # Build a simple table: role + message
            df_chat = pd.DataFrame(st.session_state.messages)
            df_chat = df_chat.rename(columns={"role": "Role", "content": "Message"})

            # --- CSV download ---
            csv_bytes = df_chat.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download chat (CSV)",
                data=csv_bytes,
                file_name="alpha_chat.csv",
                mime="text/csv",
                key="download_chat_csv",
            )

            # --- Excel download ---
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df_chat.to_excel(writer, index=False, sheet_name="Chat")
            excel_data = excel_buffer.getvalue()

            st.download_button(
                "📊 Download chat (Excel)",
                data=excel_data,
                file_name="alpha_chat.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                key="download_chat_excel",
            )



if __name__ == "__main__":
    main()
