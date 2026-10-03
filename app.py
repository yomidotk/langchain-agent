"""Streamlit UI for Focusly — Executive Study Atelier.
Complete layout overhaul: luxury light aesthetic, split studio workspace,
refined typography, tactile 3D flashcards, and full logic preservation.
"""

import asyncio
import html
import random
import re
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import aiosqlite
import streamlit as st
import streamlit.components.v1 as components
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

st.set_page_config(
    page_title="Focusly • Executive Study Atelier",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import cards_store  # noqa: E402
import chat_index  # noqa: E402
import notes_store  # noqa: E402
import tts  # noqa: E402
from agent import HERE, LANGUAGES, Context, build_agent, describe_image, secret  # noqa: E402

if not secret("DO_API_KEY"):
    st.error("Add DO_API_KEY (and ALIBABA_API_KEY) to .streamlit/secrets.toml or a .env file.")
    st.stop()


@st.cache_resource
def get_runtime():
    """Build the agent once. Async calls run on a dedicated background event loop."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    def run(coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    async def make_saver():
        return AsyncSqliteSaver(await aiosqlite.connect(str(HERE / "study_buddy.db")))

    client = MultiServerMCPClient(
        {"study_tools": {"command": sys.executable, "args": [str(HERE / "mcp_server.py")], "transport": "stdio"}}
    )
    mcp_tools = run(client.get_tools())
    agent = build_agent(mcp_tools, run(make_saver()))
    return run, agent


run, agent = get_runtime()


# ---------------------------------------------------------------- Styling
def inject_theme() -> None:
    """Inject the quiet-luxury, light-mode design system."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500;1,600&display=swap');

        :root {
            --bg-canvas: #FAF9F6;
            --bg-card: #FFFFFF;
            --bg-panel: #F7F6F1;
            --bg-hover: #F2EFE8;
            --text-heading: #16161A;
            --text-body: #2B2C33;
            --text-muted: #6E6D76;
            --text-subtle: #9896A0;
            --border-light: #EBE7DF;
            --border-hover: #D5CEBF;
            --border-focus: #18181B;
            --accent-gold: #9E7449;
            --accent-gold-bg: #F5EFE6;
            --accent-emerald: #2E6B47;
            --accent-emerald-bg: #E8F2EB;
            --shadow-subtle: 0 1px 3px rgba(0,0,0,0.02), 0 6px 20px -4px rgba(45, 38, 25, 0.04);
            --shadow-card: 0 2px 8px rgba(0,0,0,0.02), 0 14px 36px -6px rgba(45, 38, 25, 0.05);
        }

        /* App Base */
        .stApp {
            background-color: var(--bg-canvas) !important;
            color: var(--text-body) !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            -webkit-font-smoothing: antialiased;
        }

        /* Top Header */
        [data-testid="stHeader"] {
            background: rgba(250, 249, 246, 0.88) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border-bottom: 1px solid var(--border-light) !important;
        }

        .block-container {
            max-width: 1320px !important;
            padding: 1.8rem 2.4rem 4rem !important;
        }

        /* Typography */
        h1, h2, h3, h4 {
            color: var(--text-heading) !important;
            letter-spacing: -0.025em;
        }

        h1 {
            font-family: 'Playfair Display', Georgia, serif !important;
            font-weight: 600 !important;
            font-size: clamp(2rem, 3.6vw, 2.75rem) !important;
            line-height: 1.16 !important;
        }

        h2 {
            font-size: 1.25rem !important;
            font-weight: 650 !important;
        }

        /* Navbar Layout */
        .luxury-navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.2rem 0 1.4rem;
            margin-bottom: 1.4rem;
            border-bottom: 1px solid var(--border-light);
        }

        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .brand-monogram {
            display: grid;
            place-items: center;
            width: 2.35rem;
            height: 2.35rem;
            border-radius: 9px;
            background: #18181B;
            color: #FAF8F5;
            font-family: 'Playfair Display', Georgia, serif;
            font-weight: 600;
            font-style: italic;
            font-size: 1.25rem;
            box-shadow: 0 4px 14px rgba(24, 24, 27, 0.16);
        }

        .brand-text strong {
            display: block;
            color: #18181B;
            font-size: 1.18rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }

        .brand-text span {
            display: block;
            color: var(--accent-gold);
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .navbar-status {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            background: #FFFFFF;
            border: 1px solid var(--border-light);
            border-radius: 999px;
            padding: 0.4rem 0.95rem;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-muted);
            box-shadow: var(--shadow-subtle);
        }

        .status-dot {
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 50%;
            background: var(--accent-emerald);
            box-shadow: 0 0 0 3px var(--accent-emerald-bg);
        }

        /* Top Segmented Navigation */
        div[data-testid="stRadio"]:has(input[value="💬 Studio Chat"]) {
            background: #EFECE5 !important;
            border-radius: 14px !important;
            padding: 4px !important;
            display: inline-flex !important;
            width: 100% !important;
            max-width: 480px !important;
            margin: 0 auto 1.6rem !important;
        }

        div[data-testid="stRadio"]:has(input[value="💬 Studio Chat"]) label {
            flex: 1 !important;
            text-align: center !important;
            justify-content: center !important;
            padding: 0.62rem 1.1rem !important;
            border-radius: 10px !important;
            font-size: 0.92rem !important;
            font-weight: 550 !important;
            color: var(--text-muted) !important;
            transition: all 0.16s ease !important;
        }

        div[data-testid="stRadio"]:has(input[value="💬 Studio Chat"]) label:has(input:checked) {
            background: #FFFFFF !important;
            color: #18181B !important;
            font-weight: 650 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        }

        div[data-testid="stRadio"]:has(input[value="💬 Studio Chat"]) input {
            display: none !important;
        }

        /* Hero Welcome Banner */
        .hero {
            position: relative;
            padding: 2.8rem 3rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #FFFFFF 0%, #FAF8F5 55%, #F4EFE6 100%);
            border: 1px solid var(--border-light);
            box-shadow: var(--shadow-card);
            margin-bottom: 1.8rem;
        }

        .hero-kicker {
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--accent-gold);
            margin-bottom: 0.8rem;
        }

        .hero h1 {
            color: #18181B !important;
            max-width: 42rem;
            margin: 0 !important;
        }

        .hero p {
            color: #55535E;
            max-width: 38rem;
            margin: 0.95rem 0 0;
            font-size: 1.02rem;
            line-height: 1.62;
        }

        /* Starter Prompt Cards */
        .starter-heading {
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin: 1.8rem 0 0.85rem;
        }

        .starter-card .stButton > button {
            height: 7.2rem !important;
            white-space: normal !important;
            text-align: left !important;
            padding: 1.1rem 1.25rem !important;
            align-items: flex-start !important;
            background: #FFFFFF !important;
            border: 1px solid var(--border-light) !important;
            border-radius: 16px !important;
            color: #1A1A1E !important;
            box-shadow: var(--shadow-subtle) !important;
            transition: all 0.18s ease !important;
        }

        .starter-card .stButton > button:hover {
            border-color: #C5A880 !important;
            box-shadow: 0 10px 28px -4px rgba(197, 168, 128, 0.2) !important;
            transform: translateY(-2px) !important;
            color: #111 !important;
        }

        /* Executive Side Desk / Panel */
        .side-desk-box {
            background: var(--bg-panel);
            border: 1px solid var(--border-light);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: var(--shadow-subtle);
        }

        .side-desk-title {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .note-card {
            background: #FFFFFF;
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 0.75rem 0.95rem;
            margin-bottom: 0.55rem;
            font-size: 0.86rem;
            color: #2D2D33;
            display: flex;
            align-items: flex-start;
            gap: 0.55rem;
            line-height: 1.45;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }

        .note-diamond {
            color: var(--accent-gold);
            font-size: 0.75rem;
            line-height: 1.4;
            flex-shrink: 0;
        }

        /* Chat Message Bubbles */
        [data-testid="stChatMessage"] {
            background: #FFFFFF !important;
            border: 1px solid var(--border-light) !important;
            border-radius: 18px !important;
            padding: 1.25rem 1.45rem !important;
            margin-bottom: 1.05rem !important;
            box-shadow: var(--shadow-subtle) !important;
            line-height: 1.66 !important;
            color: #1A1A1F !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #F4F1EA !important;
            border-color: #E8E2D5 !important;
            border-radius: 18px 18px 4px 18px !important;
        }

        /* Floating Studio Input Bar */
        [data-testid="stChatInput"] {
            border: 1.5px solid #DFDBD2 !important;
            border-radius: 20px !important;
            background: #FFFFFF !important;
            box-shadow: 0 12px 36px -6px rgba(40, 35, 25, 0.08) !important;
            transition: all 0.16s ease !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: #18181B !important;
            box-shadow: 0 14px 40px -4px rgba(24, 24, 27, 0.12) !important;
        }

        [data-testid="stChatInput"] textarea {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 0.98rem !important;
            color: #18181B !important;
        }

        /* Standard Buttons */
        .stButton > button {
            min-height: 2.6rem;
            border-radius: 11px;
            border: 1px solid var(--border-light);
            background: #FFFFFF;
            color: #18181B;
            font-weight: 600;
            font-size: 0.88rem;
            transition: all 0.16s ease;
        }

        .stButton > button:hover {
            border-color: #18181B;
            color: #18181B;
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        }

        .stButton > button[kind="primary"] {
            background: #18181B !important;
            border: 1px solid #18181B !important;
            color: #FAF8F5 !important;
            box-shadow: 0 4px 14px rgba(24,24,27,0.14) !important;
        }

        .stButton > button[kind="primary"]:hover {
            background: #27272A !important;
            box-shadow: 0 6px 20px rgba(24,24,27,0.2) !important;
        }

        /* Form Inputs */
        div[data-baseweb="select"] > div, .stTextInput input {
            border-radius: 11px !important;
            border-color: #E0DCD3 !important;
            background: #FFFFFF !important;
            color: #18181B !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background: #FFFFFF !important;
            border: 1px solid var(--border-light) !important;
            border-radius: 16px !important;
            padding: 1.25rem 1.4rem !important;
            box-shadow: var(--shadow-subtle) !important;
        }

        [data-testid="stMetric"] label {
            color: var(--text-muted) !important;
            font-size: 0.76rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-family: 'Playfair Display', Georgia, serif !important;
            font-size: 2rem !important;
            font-weight: 600 !important;
            color: #18181B !important;
        }

        /* Action Dialog */
        .action-card {
            background: #FFFFFF;
            border: 1px solid #EAE5DB;
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 0.9rem;
            box-shadow: var(--shadow-subtle);
        }

        .action-badge {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent-gold);
            background: var(--accent-gold-bg);
            border: 1px solid #EAE3D4;
            padding: 3px 11px;
            border-radius: 999px;
            margin-bottom: 0.45rem;
        }

        /* Deck Toolbar */
        .deck-toolbar {
            background: #FFFFFF;
            border: 1px solid var(--border-light);
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            margin: 1.2rem 0;
            box-shadow: var(--shadow-subtle);
        }

        /* Dataframe */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border-light) !important;
            border-radius: 16px !important;
            overflow: hidden !important;
            background: #FFFFFF !important;
            box-shadow: var(--shadow-subtle) !important;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--border-light) !important;
            border-radius: 14px !important;
            background: #FFFFFF !important;
        }

        audio {
            height: 38px;
            border-radius: 999px;
        }

        /* Clean Sidebar (Collapsed by default, styled if toggled) */
        [data-testid="stSidebar"] {
            background-color: var(--bg-panel) !important;
            border-right: 1px solid #EAE5DC !important;
        }

        [data-testid="stSidebar"] * {
            color: #18181B !important;
        }

        @media (max-width: 900px) {
            .block-container { padding: 1.2rem 1.1rem 3rem !important; }
            .hero { padding: 2rem 1.6rem; }
            .luxury-navbar { flex-direction: column; align-items: flex-start; gap: 0.8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

# ---------------------------------------------------------------- State & Routing
ss = st.session_state
ss.setdefault("study", None)  # active flashcard session
ss.setdefault("playing", None)  # audio id currently speaking

# Session recovery: URL query param (?chat=...) wins, then chats.json, then new UUID
thread_id = st.query_params.get("chat") or chat_index.load()["last"] or chat_index.new_id()
st.query_params["chat"] = thread_id
chat_index.set_last(thread_id)
config = {"configurable": {"thread_id": thread_id}}


def new_conversation():
    st.query_params["chat"] = chat_index.new_id()


def open_chat(chat_id: str):
    st.query_params["chat"] = chat_id


# ---------------------------------------------------------------- Top Navigation
st.markdown(
    """
    <header class="luxury-navbar">
        <div class="navbar-brand">
            <div class="brand-monogram">✦</div>
            <div class="brand-text">
                <strong>Focusly</strong>
                <span>Executive Study Atelier</span>
            </div>
        </div>
        <div class="navbar-status">
            <span class="status-dot"></span>
            Personal AI Concierge Active
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)

# Centered luxury navigation pill
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
with nav_col2:
    page = st.radio(
        "Navigation",
        ["💬 Studio Chat", "🃏 Knowledge Decks"],
        horizontal=True,
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------- Sidebar Fallback
# Clean secondary controls if user chooses to open the sidebar
with st.sidebar:
    st.markdown("<div style='font-size:0.75rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#8A8780;margin-bottom:0.8rem;'>Study Profile</div>", unsafe_allow_html=True)
    sb_user_name = st.text_input("Name", value="Chiraz", key="sb_name")
    sb_level = st.selectbox("Proficiency", ["beginner", "intermediate", "advanced"], index=1, key="sb_level")
    sb_language = st.selectbox("Language", list(LANGUAGES), key="sb_lang")
    has_tts = bool(secret("ALIBABA_API_KEY"))
    sb_voice, sb_style = "Cherry", "Default voice"
    if has_tts:
        sb_voice = st.selectbox("Voice", tts.VOICES, key="sb_voice")
        sb_style = st.selectbox("Style", list(tts.STYLES), key="sb_style")
    st.button("✦ Start New Conversation", on_click=new_conversation, type="primary", use_container_width=True, key="sb_new_chat")

# Defaults from profile
user_name = sb_user_name
level = sb_level
language = sb_language
voice = sb_voice
style = sb_style

context = Context(user_name=user_name, level=level, language=language)
state = run(agent.aget_state(config))
values = state.values if state and state.values else {}


# =================================================================== Chat
IMAGE_TAIL = re.compile(r"\n\n\[Attached image, described by a vision model:.*\]\s*$", re.DOTALL)


def history(messages):
    """Format LangGraph state messages into clean (role, content) pairs."""
    out = []
    for m in messages:
        if m.additional_kwargs.get("lc_source") == "summarization":
            continue
        if m.type == "human":
            out.append(("user", IMAGE_TAIL.sub("\n\n🖼️ *image attached*", m.text)))
        elif m.type == "ai" and m.text.strip():
            out.append(("assistant", m.text))
    return out


def mostly_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum("\u0600" <= c <= "\u06ff" for c in letters) / len(letters) > 0.3


def listen_ui(i: int, text: str):
    """Audio player for speech synthesis."""
    if language.startswith("Darija") or mostly_arabic(text):
        st.caption("🔇 Voice generation is optimized for English and Français.")
        return
    key = f"{thread_id}_{i}"
    fresh = st.button("🔊 Listen to response", key=f"listen_{key}")
    if fresh:
        ss.playing = key
    if ss.playing == key:
        try:
            with st.spinner("Synthesizing voice..."):
                audio = tts.synthesize(text, secret("ALIBABA_API_KEY"), voice, style)
            st.audio(audio, format="audio/wav", autoplay=fresh)
        except Exception as e:
            st.error(f"Couldn't generate audio: {e}")


def call_agent(payload):
    with st.spinner("Refining response..."):
        run(agent.ainvoke(payload, config=config, context=context))


def chat_page():
    messages = history(values.get("messages", []))

    # Modern 2-column split workspace: Main Chat Stage + Executive Side Desk
    chat_stage_col, side_desk_col = st.columns([13, 7], gap="large")

    with side_desk_col:
        # Executive Side Desk Box
        st.markdown(
            """
            <div class="side-desk-box">
                <div class="side-desk-title">
                    <span>✦ Executive Side Desk</span>
                    <span>Studio Hub</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Profile Accordion
        with st.expander("👤 Learning Profile & Voice", expanded=False):
            p_name = st.text_input("Student Name", value=user_name, key="desk_name")
            p_level = st.selectbox("Proficiency Level", ["beginner", "intermediate", "advanced"], index=["beginner", "intermediate", "advanced"].index(level), key="desk_level")
            p_lang = st.selectbox("Instruction Language", list(LANGUAGES), index=list(LANGUAGES).index(language), key="desk_lang")
            if has_tts:
                p_voice = st.selectbox("Preferred Voice", tts.VOICES, index=tts.VOICES.index(voice) if voice in tts.VOICES else 0, key="desk_voice")
                p_style = st.selectbox("Delivery Style", list(tts.STYLES), index=list(tts.STYLES).index(style) if style in tts.STYLES else 0, key="desk_style")

        # Study Notes
        st.markdown("<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#8A8780;margin:1.2rem 0 0.6rem;'>📝 Active Study Memos</div>", unsafe_allow_html=True)
        saved_notes = notes_store.load()
        if saved_notes:
            for n in saved_notes[-6:]:  # show most recent
                safe_n = html.escape(n)
                st.markdown(f"<div class='note-card'><span class='note-diamond'>✦</span><span>{safe_n}</span></div>", unsafe_allow_html=True)
        else:
            st.caption("Ask the tutor to save a key takeaway anytime.")

        # Past Conversations List
        st.markdown("<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#8A8780;margin:1.4rem 0 0.6rem;'>🗂️ Session Archive</div>", unsafe_allow_html=True)
        st.button("✦ Start New Session", on_click=new_conversation, type="primary", use_container_width=True, key="desk_new_chat")
        past = chat_index.recent()
        if past:
            for cid, title in past[:5]:
                is_active = (cid == thread_id)
                prefix = "● " if is_active else ""
                st.button(f"{prefix}{title}", key=f"d_chat_{cid}", on_click=open_chat, args=(cid,), disabled=is_active, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with chat_stage_col:
        quick_prompt = None

        # Empty Chat: Luxury Greeting & 3 Curated Prompt Cards
        if not messages:
            safe_name = html.escape(user_name or "there")
            st.markdown(
                f"""
                <section class="hero">
                    <div class="hero-kicker">✦ Personalized Learning Studio</div>
                    <h1>Make your next study session count, {safe_name}.</h1>
                    <p>Explore ideas with clarity, master challenging concepts through custom flashcards, or inspect diagrams with vision AI.</p>
                </section>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div class='starter-heading'>Guided Study Accelerators</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("<div class='starter-card'>", unsafe_allow_html=True)
                if st.button("💡 Explain simply\nBreak down complex ideas into intuitive, memorable concepts", key="s_explain", use_container_width=True):
                    quick_prompt = "Explain a difficult topic to me in simple terms."
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='starter-card'>", unsafe_allow_html=True)
                if st.button("📋 Revision plan\nBuild a structured, milestone-based study routine", key="s_plan", use_container_width=True):
                    quick_prompt = "Create a focused revision plan for me."
                st.markdown("</div>", unsafe_allow_html=True)
            with c3:
                st.markdown("<div class='starter-card'>", unsafe_allow_html=True)
                if st.button("🃏 Flashcard deck\nTurn what you're learning into active recall cards", key="s_cards", use_container_width=True):
                    quick_prompt = "Make me flashcards for a topic I am studying."
                st.markdown("</div>", unsafe_allow_html=True)

        # Message Bubbles
        for i, (role, text) in enumerate(messages):
            with st.chat_message(role):
                st.markdown(text)
                if role == "assistant" and has_tts:
                    listen_ui(i, text)

        # Human-in-the-Loop Dialog
        pending = None
        for it in getattr(state, "interrupts", ()) or ():
            pending = it.value["action_requests"]
            break

        if pending:
            with st.chat_message("assistant"):
                st.markdown(
                    """
                    <div class='action-card'>
                        <span class='action-badge'>Authorization Required</span>
                        <h4 style='margin:0.2rem 0 0.4rem;font-size:1.05rem;color:#18181B;font-weight:650;'>Permission requested to execute an action</h4>
                        <p style='margin:0;color:#5E5D66;font-size:0.9rem;'>The tutor needs your confirmation before continuing with this operation:</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for r in pending:
                    action_name = r.get("name", "Action")
                    action_args = r.get("args", {})
                    arg_desc = ", ".join(f"{k}: {v}" for k, v in action_args.items()) if isinstance(action_args, dict) else str(action_args)
                    st.markdown(
                        f"<div style='background:#F8F7F4;border:1px solid #EAE5DC;border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.75rem;font-size:0.88rem;color:#1C1D22;'>"
                        f"<strong style='color:#18181B;margin-right:0.6rem;'>✦ {html.escape(action_name)}</strong> <span style='color:#55535E;'>{html.escape(arg_desc)}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                col1, col2 = st.columns(2)
                decision = None
                if col1.button("✅ Authorize Action", type="primary", use_container_width=True, key="hitl_app"):
                    decision = {"type": "approve"}
                if col2.button("❌ Decline", use_container_width=True, key="hitl_dec"):
                    decision = {"type": "reject", "message": "User said no."}
            if decision:
                call_agent(Command(resume={"decisions": [decision] * len(pending)}))
                st.rerun()

        # Chat Input
        prompt = st.chat_input(
            "Ask a question, request a concept breakdown, or attach an image...",
            accept_file=True,
            file_type=["png", "jpg", "jpeg"],
            disabled=bool(pending),
        )

        if prompt or quick_prompt:
            text = quick_prompt or prompt.text or "Describe this image."
            to_agent = text
            chat_index.touch(thread_id, title=text)
            with st.chat_message("user"):
                st.markdown(text)
            if prompt and prompt.files:
                f = prompt.files[0]
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                    tmp.write(f.getvalue())
                with st.spinner("Analyzing diagram with vision model..."):
                    description = run(describe_image(tmp.name, text))
                to_agent = f"{text}\n\n[Attached image, described by a vision model: {description}]"
            call_agent({"messages": [{"role": "user", "content": to_agent}]})
            st.rerun()


# ============================================================== Flashcards
def show_html(code: str, height: int) -> None:
    if hasattr(st, "iframe"):
        st.iframe(code, height=height)
    else:
        components.html(code, height=height)


def flip_card(concept: str, question: str, answer: str) -> None:
    """Tactile heavy-linen 3D flip card with luxury styling."""
    c, q, a = html.escape(concept), html.escape(question), html.escape(answer)
    show_html(
        f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;1,500&display=swap');
  body {{ margin: 0; background: transparent; font-family: 'Plus Jakarta Sans', -apple-system, sans-serif; }}
  .scene {{ perspective: 1200px; width: 100%; max-width: 620px; height: 300px; margin: 0 auto; animation: pop .4s cubic-bezier(0.16, 1, 0.3, 1); }}
  .card {{ position: relative; width: 100%; height: 100%; cursor: pointer; transition: transform .65s cubic-bezier(.4,.2,.2,1); transform-style: preserve-3d; }}
  .card.flipped {{ transform: rotateY(180deg); }}
  .face {{ position: absolute; inset: 0; backface-visibility: hidden; -webkit-backface-visibility: hidden; border-radius: 24px; padding: 32px 38px; box-sizing: border-box; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
  
  .front {{ 
    background: linear-gradient(145deg, #FFFFFF 0%, #FAF8F5 100%); 
    border: 1.5px solid #E6E1D6; 
    box-shadow: 0 18px 44px -10px rgba(50, 45, 35, 0.08), 0 2px 6px rgba(0,0,0,0.02); 
    color: #1A1A1E; 
  }}
  .front .text {{ font-family: 'Playfair Display', Georgia, serif; font-size: 24px; font-weight: 550; font-style: italic; line-height: 1.42; color: #18181B; max-height: 170px; overflow-y: auto; }}
  .front .tag {{ background: #F3EFE8; color: #8A6538; border: 1px solid #E5DFD4; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; font-weight: 700; border-radius: 999px; padding: 5px 14px; margin-bottom: 18px; }}
  .front .hint {{ position: absolute; bottom: 15px; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #9A958A; font-weight: 600; }}

  .back {{ 
    background: linear-gradient(145deg, #FCFDFC 0%, #F1F6F2 100%); 
    border: 1.5px solid #D6E4D8; 
    box-shadow: 0 18px 44px -10px rgba(35, 50, 40, 0.08), 0 2px 6px rgba(0,0,0,0.02); 
    color: #1A281F; 
    transform: rotateY(180deg); 
  }}
  .back .text {{ font-family: 'Plus Jakarta Sans', sans-serif; font-size: 20px; font-weight: 550; line-height: 1.48; color: #18281F; max-height: 170px; overflow-y: auto; }}
  .back .tag {{ background: #E5EFE6; color: #2E6B47; border: 1px solid #D0E2D3; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; font-weight: 700; border-radius: 999px; padding: 5px 14px; margin-bottom: 18px; }}
  .back .hint {{ position: absolute; bottom: 15px; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #728A7A; font-weight: 600; }}

  @keyframes pop {{ from {{ opacity: 0; transform: translateY(14px) scale(.98); }} to {{ opacity: 1; transform: none; }} }}
</style>
<div class="scene">
  <div class="card" onclick="this.classList.toggle('flipped')">
    <div class="face front">
      <div class="tag">{c} &middot; Question</div>
      <div class="text">{q}</div>
      <div class="hint">Click to reveal answer ↺</div>
    </div>
    <div class="face back">
      <div class="tag">Answer</div>
      <div class="text">{a}</div>
      <div class="hint">Click to flip back ↺</div>
    </div>
  </div>
</div>
""".strip(),
        height=320,
    )


def start_study(concept: str, ids: list[str]) -> None:
    ids = list(ids)
    random.shuffle(ids)
    ss.study = {"concept": concept, "queue": ids, "pos": 0, "results": {}}


def stop_study() -> None:
    ss.study = None


def grade(correct: bool, card_id: str) -> None:
    study = ss.study
    cards_store.record(study["concept"], card_id, correct)
    study["results"][card_id] = correct
    study["pos"] += 1


def overview(decks: dict) -> None:
    rows = []
    for concept, cards in decks.items():
        right = sum(c["right"] for c in cards)
        tries = right + sum(c["wrong"] for c in cards)
        rows.append(
            {
                "Concept": concept,
                "Cards": len(cards),
                "Got it ✅": sum(c["last"] == "right" for c in cards),
                "To redo ❌": sum(c["last"] == "wrong" for c in cards),
                "New 🆕": sum(c["last"] is None for c in cards),
                "Accuracy": f"{right / tries:.0%}" if tries else "-",
            }
        )
    total_cards = sum(len(cards) for cards in decks.values())
    reviewed = sum(sum(card["last"] is not None for card in cards) for cards in decks.values())
    
    # 4 Luxury Stat Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Study Decks", len(decks))
    m2.metric("Saved Cards", total_cards)
    m3.metric("Cards Reviewed", reviewed)
    overall_right = sum(sum(c["right"] for c in cards) for cards in decks.values())
    overall_tries = overall_right + sum(sum(c["wrong"] for c in cards) for cards in decks.values())
    m4.metric("Overall Mastery", f"{overall_right / overall_tries:.0%}" if overall_tries else "—")

    st.markdown("<div class='deck-toolbar'><h3>Select Deck to Practice</h3><p>Choose a concept below and target all cards or focus specifically on missed items.</p></div>", unsafe_allow_html=True)
    st.dataframe(rows, hide_index=True, use_container_width=True)

    concept = st.selectbox("Pick a concept to study", list(decks))
    cards = decks[concept]
    missed = [c["id"] for c in cards if c["last"] == "wrong"]
    mode = st.radio("Practice Target", ["All cards", f"Only the ones I missed ({len(missed)})"], horizontal=True)
    only_missed = mode.startswith("Only")
    
    if only_missed and not missed:
        st.info("Nothing missed here, nice! Study all cards instead.")
    elif st.button("▶️ Launch Study Session", type="primary"):
        start_study(concept, missed if only_missed else [c["id"] for c in cards])
        st.rerun()

    with st.expander("Manage decks"):
        if st.button(f"🗑️ Delete the '{concept}' deck"):
            cards_store.delete_deck(concept)
            st.rerun()


def study_view(decks: dict) -> None:
    study = ss.study
    concept = study["concept"]
    cards = {c["id"]: c for c in decks.get(concept, [])}
    queue = [i for i in study["queue"] if i in cards]
    pos, total = study["pos"], len(queue)

    st.button("← Back to Decks Overview", on_click=stop_study)

    if pos < total:
        card = cards[queue[pos]]
        st.progress(pos / total, text=f"{concept}: Card {pos + 1} of {total}")
        flip_card(concept, card["question"], card["answer"])
        
        # Tactile grading controls
        g1, g2, g3 = st.columns([1, 2, 1])
        with g2:
            col1, col2 = st.columns(2)
            col1.button("✅ I Got It", on_click=grade, args=(True, card["id"]), use_container_width=True)
            col2.button("❌ Missed It", on_click=grade, args=(False, card["id"]), use_container_width=True)
        return

    # Completion score
    results = study["results"]
    right = sum(results.values())
    st.markdown(f"<div style='text-align:center;margin:1.8rem 0;'><h2>🎉 Session Completed for {html.escape(concept)}</h2></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Mastered", right)
    c2.metric("Missed", total - right)
    c3.metric("Session Accuracy", f"{right / total:.0%}" if total else "-")
    if total and right == total:
        st.balloons()
    missed_ids = [i for i, ok in results.items() if not ok]
    if missed_ids:
        with st.expander(f"Review the {len(missed_ids)} cards you missed"):
            for i in missed_ids:
                st.markdown(f"**{cards[i]['question']}**  \n{cards[i]['answer']}")
        if st.button("🔁 Re-practice Missed Cards", type="primary"):
            start_study(concept, missed_ids)
            st.rerun()
    if st.button("🔄 Study the Entire Deck Again"):
        start_study(concept, list(cards))
        st.rerun()


def flashcards_page():
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">✦ Spaced Repetition Studio</div>
            <h1>Build knowledge that endures.</h1>
            <p>Master complex topics through high-retention 3D flashcards. Review full decks or focus specifically on cards you missed.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    decks = cards_store.load()
    if not decks:
        st.info("No flashcards yet. Go to the Studio Chat and ask: *make me flashcards about short-term memory*.")
        return
    if ss.study and ss.study["concept"] in decks:
        study_view(decks)
    else:
        ss.study = None
        overview(decks)


# Page Routing
if page == "💬 Studio Chat":
    chat_page()
else:
    flashcards_page()
