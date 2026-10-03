"""Streamlit UI for Focusly (chat + flip-card flashcards). Run with: streamlit run app.py"""

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

st.set_page_config(page_title="Focusly", page_icon="*", layout="wide", initial_sidebar_state="expanded")

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
    """Build the agent once. Async calls run on one background event loop, because
    Streamlit reruns the script on every click and async HTTP clients dislike changing loops."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    def run(coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    async def make_saver():  # short-term memory saved in a SQLite file, so it survives refreshes and restarts
        return AsyncSqliteSaver(await aiosqlite.connect(str(HERE / "study_buddy.db")))

    client = MultiServerMCPClient(
        {"study_tools": {"command": sys.executable, "args": [str(HERE / "mcp_server.py")], "transport": "stdio"}}
    )
    mcp_tools = run(client.get_tools())  # MCP
    agent = build_agent(mcp_tools, run(make_saver()))
    return run, agent


run, agent = get_runtime()

def inject_theme() -> None:
    """Apply the luxury, light-mode editorial interface for Focusly."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500;1,600&display=swap');

        :root {
            --bg-app: #FAF9F6;
            --bg-card: #FFFFFF;
            --bg-sidebar: #F7F6F2;
            --text-main: #18181B;
            --text-muted: #6E6D76;
            --text-subtle: #9896A0;
            --border-light: #EBE8E1;
            --border-hover: #D4CEBF;
            --border-focus: #18181B;
            --accent-gold: #9E7449;
            --accent-gold-pale: #F3EFE8;
            --accent-green: #2E6B47;
            --shadow-subtle: 0 1px 3px rgba(0,0,0,0.02), 0 6px 18px -4px rgba(40, 35, 25, 0.04);
            --shadow-card: 0 2px 8px rgba(0,0,0,0.02), 0 14px 34px -6px rgba(40, 35, 25, 0.05);
        }

        /* App Canvas */
        .stApp {
            background-color: var(--bg-app) !important;
            color: var(--text-main) !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            -webkit-font-smoothing: antialiased;
        }

        /* App Header */
        [data-testid="stHeader"] {
            background: rgba(250, 249, 246, 0.88) !important;
            backdrop-filter: blur(14px) !important;
            -webkit-backdrop-filter: blur(14px) !important;
            border-bottom: 1px solid var(--border-light) !important;
        }

        /* Sidebar Styling - Light, Airy, Luxury */
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid #E8E5DC !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 1.5rem 1.15rem 2rem !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text-main);
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: var(--text-subtle) !important;
            font-size: 0.82rem !important;
        }

        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stSelectbox label {
            color: #55545B !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stTextInput input {
            background: #FFFFFF !important;
            border: 1px solid #DFDBD2 !important;
            color: #18181B !important;
            border-radius: 10px !important;
            font-size: 0.88rem !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
            transition: all 0.15s ease !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div:hover,
        [data-testid="stSidebar"] .stTextInput input:hover {
            border-color: #C0BCB3 !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within,
        [data-testid="stSidebar"] .stTextInput input:focus {
            border-color: #18181B !important;
            box-shadow: 0 0 0 2px rgba(24, 24, 27, 0.08) !important;
        }

        /* Sidebar Buttons */
        [data-testid="stSidebar"] .stButton > button {
            background: #FFFFFF !important;
            border: 1px solid #E8E4DB !important;
            color: #2D2D33 !important;
            text-align: left !important;
            border-radius: 10px !important;
            padding: 0.58rem 0.8rem !important;
            font-size: 0.86rem !important;
            font-weight: 550 !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
            transition: all 0.16s ease !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: #F3EFE7 !important;
            border-color: #D8D2C4 !important;
            color: #111111 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.04) !important;
        }

        [data-testid="stSidebar"] .stButton > button:disabled {
            background: #EFECE4 !important;
            border-color: #D6CFBF !important;
            color: #18181B !important;
            font-weight: 650 !important;
            opacity: 1 !important;
        }

        /* New Conversation Button */
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: #18181B !important;
            border: 1px solid #27272A !important;
            color: #FAF8F5 !important;
            text-align: center !important;
            border-radius: 10px !important;
            padding: 0.65rem 0.9rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 14px rgba(24,24,27,0.12) !important;
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            background: #27272A !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 18px rgba(24,24,27,0.18) !important;
        }

        /* Sidebar Radio Pills */
        [data-testid="stSidebar"] [data-testid="stRadio"] {
            background: #EFECE5;
            border-radius: 12px;
            padding: 4px;
            gap: 2px;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            width: 100%;
            padding: 0.55rem 0.8rem;
            border-radius: 8px;
            font-size: 0.88rem;
            font-weight: 500;
            color: #55545B;
            transition: all 0.15s ease;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: #FFFFFF !important;
            color: #18181B !important;
            font-weight: 650 !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] input {
            display: none;
        }

        /* Typography & Headings */
        h1, h2, h3 {
            color: var(--text-main) !important;
            letter-spacing: -0.02em;
        }

        h1 {
            font-family: 'Playfair Display', Georgia, serif !important;
            font-weight: 600 !important;
            font-size: clamp(2rem, 3.8vw, 2.9rem) !important;
            line-height: 1.15 !important;
        }

        h2 {
            font-size: 1.28rem !important;
            font-weight: 650 !important;
            margin-top: 1.6rem !important;
        }

        .block-container {
            max-width: 1080px;
            padding: 2.2rem 2.4rem 4rem;
        }

        /* Brand Emblem */
        .focusly-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding: 0.2rem 0.2rem 1.6rem;
            border-bottom: 1px solid #EAE6DD;
            margin-bottom: 1rem;
        }

        .focusly-mark {
            display: grid;
            place-items: center;
            width: 2.3rem;
            height: 2.3rem;
            border-radius: 9px;
            background: #18181B;
            color: #F8F5EE;
            font-family: 'Playfair Display', Georgia, serif;
            font-weight: 600;
            font-style: italic;
            font-size: 1.2rem;
            box-shadow: 0 4px 12px rgba(24, 24, 27, 0.15);
        }

        .focusly-brand strong {
            display: block;
            color: #18181B;
            font-size: 1.18rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .focusly-brand span {
            display: block;
            color: #8C8A84;
            font-size: 0.72rem;
            letter-spacing: 0.02em;
        }

        .sidebar-section {
            color: #828078;
            font-size: 0.69rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            padding: 0.95rem 0.3rem 0.4rem;
        }

        .workspace-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.6rem;
            padding-bottom: 0.7rem;
            border-bottom: 1px solid var(--border-light);
        }

        .workspace-label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-muted);
            font-size: 0.82rem;
            font-weight: 600;
        }

        .status-dot {
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 50%;
            background: #2E6B47;
            box-shadow: 0 0 0 3px #E5EFE6;
        }

        /* Hero Card - Editorial Luxury */
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.6rem 2.8rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #FFFFFF 0%, #FAF8F5 55%, #F4EFE6 100%);
            border: 1px solid #EAE4D8;
            box-shadow: var(--shadow-card);
            margin-bottom: 1.6rem;
        }

        .hero-kicker {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--accent-gold);
            margin-bottom: 0.7rem;
        }

        .hero h1 {
            color: #18181B !important;
            max-width: 38rem;
            margin: 0 !important;
        }

        .hero p {
            color: #55535E;
            max-width: 35rem;
            margin: 0.9rem 0 0;
            font-size: 0.98rem;
            line-height: 1.62;
        }

        /* Starter Prompt Cards */
        .starter-heading {
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin: 1.5rem 0 0.75rem;
        }

        .starter-card .stButton > button {
            height: 6.8rem !important;
            white-space: normal !important;
            text-align: left !important;
            padding: 1rem 1.15rem !important;
            align-items: flex-start !important;
            background: #FFFFFF !important;
            border: 1px solid var(--border-light) !important;
            border-radius: 14px !important;
            color: #1A1A1E !important;
            box-shadow: var(--shadow-subtle) !important;
            transition: all 0.18s ease !important;
        }

        .starter-card .stButton > button:hover {
            border-color: #C5A880 !important;
            box-shadow: 0 8px 24px -4px rgba(197, 168, 128, 0.18) !important;
            transform: translateY(-2px) !important;
            color: #111 !important;
        }

        /* General Buttons */
        .stButton > button {
            min-height: 2.5rem;
            border-radius: 10px;
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
        }

        .stButton > button[kind="primary"]:hover {
            background: #27272A !important;
            box-shadow: 0 6px 18px rgba(24,24,27,0.12) !important;
        }

        /* Chat Messages */
        [data-testid="stChatMessage"] {
            background: #FFFFFF;
            border: 1px solid var(--border-light);
            border-radius: 18px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 0.95rem;
            box-shadow: var(--shadow-subtle);
            line-height: 1.65;
            color: #1C1D22;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #F4F1EA !important;
            border-color: #E8E2D5 !important;
            border-radius: 18px 18px 4px 18px !important;
        }

        /* Chat Input */
        [data-testid="stChatInput"] {
            border: 1.5px solid #DFDBD2 !important;
            border-radius: 18px !important;
            background: #FFFFFF !important;
            box-shadow: 0 10px 32px -6px rgba(40, 35, 25, 0.07) !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: #18181B !important;
            box-shadow: 0 12px 36px -4px rgba(24, 24, 27, 0.1) !important;
        }

        [data-testid="stChatInput"] textarea {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 0.96rem !important;
            color: #18181B !important;
        }

        /* Form inputs & selectboxes on page */
        div[data-baseweb="select"] > div, .stTextInput input {
            border-radius: 10px !important;
            border-color: #E0DCD3 !important;
            background: #FFFFFF !important;
        }

        /* Metrics Cards */
        [data-testid="stMetric"] {
            background: #FFFFFF !important;
            border: 1px solid var(--border-light) !important;
            border-radius: 14px !important;
            padding: 1.1rem 1.25rem !important;
            box-shadow: var(--shadow-subtle) !important;
        }

        [data-testid="stMetric"] label {
            color: var(--text-muted) !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-family: 'Playfair Display', Georgia, serif !important;
            font-size: 1.85rem !important;
            font-weight: 600 !important;
            color: #18181B !important;
        }

        /* Dataframe & Expanders */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border-light) !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            background: #FFFFFF !important;
            box-shadow: var(--shadow-subtle) !important;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--border-light) !important;
            border-radius: 12px !important;
            background: #FFFFFF !important;
        }

        .deck-toolbar {
            background: #FFFFFF;
            border: 1px solid var(--border-light);
            border-radius: 16px;
            padding: 1.35rem 1.5rem;
            margin: 1.1rem 0 1.25rem;
            box-shadow: var(--shadow-subtle);
        }

        .deck-toolbar h3 {
            margin: 0 0 0.25rem;
            font-size: 1.12rem;
            font-weight: 650;
        }

        .deck-toolbar p {
            color: var(--text-muted);
            margin: 0;
            font-size: 0.88rem;
        }

        /* Note Cards in Sidebar */
        .note-card {
            background: #FFFFFF;
            border: 1px solid #EBE7DF;
            border-radius: 10px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.45rem;
            font-size: 0.84rem;
            color: #2D2D33;
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            line-height: 1.45;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }

        .note-diamond {
            color: var(--accent-gold);
            font-size: 0.72rem;
            line-height: 1.4;
            flex-shrink: 0;
        }

        /* Action Authorization Dialog */
        .action-card {
            background: #FFFFFF;
            border: 1px solid #EAE5DB;
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.8rem;
            box-shadow: var(--shadow-subtle);
        }

        .action-badge {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #9E7449;
            background: #F8F5EE;
            border: 1px solid #EAE3D4;
            padding: 3px 10px;
            border-radius: 999px;
            margin-bottom: 0.4rem;
        }

        /* Audio Player */
        audio {
            height: 38px;
            border-radius: 999px;
        }

        @media (max-width: 700px) {
            .block-container { padding: 1.4rem 1.1rem 2.5rem; }
            .hero { padding: 1.8rem 1.5rem; }
            .workspace-top { margin-bottom: 1.2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("study", None)  # current flashcard session
ss.setdefault("playing", None)  # which answer is being read aloud

# Which chat are we in? The URL (?chat=...) wins; otherwise reopen the chat you used last.
# (chats.json remembers it on disk, so even a plain refresh or a bare localhost URL brings it back.)
thread_id = st.query_params.get("chat") or chat_index.load()["last"] or chat_index.new_id()
st.query_params["chat"] = thread_id
chat_index.set_last(thread_id)
config = {"configurable": {"thread_id": thread_id}}


def new_conversation():
    st.query_params["chat"] = chat_index.new_id()


def open_chat(chat_id: str):
    st.query_params["chat"] = chat_id


# -------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("<div class='focusly-brand'><div class='focusly-mark'>F</div><div><strong>Focusly</strong><span>Executive Study Atelier</span></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>Workspace</div>", unsafe_allow_html=True)
    page = st.radio("Page", ["Chat", "Flashcards"], label_visibility="collapsed")
    st.markdown("<div class='sidebar-section'>Study profile</div>", unsafe_allow_html=True)
    user_name = st.text_input("Your name", "Chiraz")
    level = st.selectbox("Your level", ["beginner", "intermediate", "advanced"], index=1)
    language = st.selectbox("Answer language", list(LANGUAGES))
    has_tts = bool(secret("ALIBABA_API_KEY"))
    voice, style = "Cherry", "Default voice"
    if has_tts:
        st.subheader("🔊 Voice")
        voice = st.selectbox("Voice", tts.VOICES)
        style = st.selectbox("Speaking style", list(tts.STYLES))
    st.markdown("<div class='sidebar-section'>Conversations</div>", unsafe_allow_html=True)
    st.button("New conversation", on_click=new_conversation, type="primary", use_container_width=True)
    past = chat_index.recent()
    if past:
        st.caption("Your chats")
        for cid, title in past:
            st.button(title, key=f"chat_{cid}", on_click=open_chat, args=(cid,), disabled=cid == thread_id, use_container_width=True)

context = Context(user_name=user_name, level=level, language=language)  # runtime context
state = run(agent.aget_state(config))  # everything saved for this chat: messages + notes
values = state.values if state and state.values else {}

with st.sidebar:
    st.markdown("<div class='sidebar-section'>Study notes</div>", unsafe_allow_html=True)
    saved_notes = notes_store.load()
    if saved_notes:
        for n in saved_notes:
            safe_n = html.escape(n)
            st.markdown(f"<div class='note-card'><span class='note-diamond'>✦</span><span>{safe_n}</span></div>", unsafe_allow_html=True)
    else:
        st.caption("Ask the tutor to save a note anytime.")


# =================================================================== chat
IMAGE_TAIL = re.compile(r"\n\n\[Attached image, described by a vision model:.*\]\s*$", re.DOTALL)


def history(messages):
    """Turn the saved agent messages into chat bubbles (skip tool calls and summaries)."""
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
    """A 🔊 Listen button under an answer. Click it to hear the answer instead of reading it."""
    if language.startswith("Darija") or mostly_arabic(text):  # Qwen-TTS has no Darija/Arabic voice
        st.caption("🔇 Voice isn't available for Darija yet. Switch the answer language to English or Français to listen.")
        return
    key = f"{thread_id}_{i}"
    fresh = st.button("🔊 Listen", key=f"listen_{key}")
    if fresh:
        ss.playing = key
    if ss.playing == key:
        try:
            with st.spinner("Generating the voice..."):
                audio = tts.synthesize(text, secret("ALIBABA_API_KEY"), voice, style)
            st.audio(audio, format="audio/wav", autoplay=fresh)
        except Exception as e:
            st.error(f"Couldn't generate the voice: {e}")


def call_agent(payload):
    with st.spinner("Thinking..."):
        run(agent.ainvoke(payload, config=config, context=context))


def chat_page():
    messages = history(values.get("messages", []))
    st.markdown("<div class='workspace-top'><div class='workspace-label'><span class='status-dot'></span> Study Atelier Active</div><div class='workspace-label'>Personalized Learning Studio</div></div>", unsafe_allow_html=True)
    quick_prompt = None
    if not messages:
        safe_name = html.escape(user_name or "there")
        st.markdown(f"<section class='hero'><div class='hero-kicker'>✦ Personalized Learning Studio</div><h1>Make your next study session count, {safe_name}.</h1><p>Explore ideas with clarity, master challenging concepts through custom flashcards, or inspect diagrams with vision AI.</p></section>", unsafe_allow_html=True)
        st.markdown("<div class='starter-heading'>Guided study prompts</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='starter-card'>", unsafe_allow_html=True)
            if st.button("💡 Explain simply\nBreak down complex ideas into intuitive, memorable concepts", key="starter_explain", use_container_width=True):
                quick_prompt = "Explain a difficult topic to me in simple terms."
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='starter-card'>", unsafe_allow_html=True)
            if st.button("📋 Revision plan\nBuild a structured, milestone-based study routine", key="starter_plan", use_container_width=True):
                quick_prompt = "Create a focused revision plan for me."
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='starter-card'>", unsafe_allow_html=True)
            if st.button("🃏 Flashcard deck\nTurn what you're learning into active recall cards", key="starter_cards", use_container_width=True):
                quick_prompt = "Make me flashcards for a topic I am studying."
            st.markdown("</div>", unsafe_allow_html=True)

    for i, (role, text) in enumerate(messages):
        with st.chat_message(role):
            st.markdown(text)
            if role == "assistant" and has_tts:
                listen_ui(i, text)

    pending = None  # the agent paused and needs a human decision
    for it in getattr(state, "interrupts", ()) or ():
        pending = it.value["action_requests"]
        break

    if pending:
        with st.chat_message("assistant"):
            st.markdown(
                """
                <div class='action-card'>
                    <span class='action-badge'>Authorization Required</span>
                    <h4 style='margin:0.2rem 0 0.4rem;font-size:1.05rem;color:#18181B;font-weight:650;'>Permission requested to perform an external action</h4>
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
            if col1.button("✅ Authorize Action", type="primary", use_container_width=True):
                decision = {"type": "approve"}
            if col2.button("❌ Decline", use_container_width=True):
                decision = {"type": "reject", "message": "User said no."}
        if decision:
            call_agent(Command(resume={"decisions": [decision] * len(pending)}))
            st.rerun()

    prompt = st.chat_input(
        "Ask me anything, or attach an image...",
        accept_file=True,
        file_type=["png", "jpg", "jpeg"],
        disabled=bool(pending),
    )

    if prompt or quick_prompt:
        text = quick_prompt or prompt.text or "Describe this image."
        to_agent = text
        chat_index.touch(thread_id, title=text)  # registers the chat in your list on its first message
        with st.chat_message("user"):
            st.markdown(text)
        if prompt and prompt.files:  # multimodal: a vision model describes the image, the agent gets the description
            f = prompt.files[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                tmp.write(f.getvalue())
            with st.spinner("Looking at the image..."):
                description = run(describe_image(tmp.name, text))
            to_agent = f"{text}\n\n[Attached image, described by a vision model: {description}]"
        call_agent({"messages": [{"role": "user", "content": to_agent}]})
        st.rerun()


# ============================================================== flashcards
def show_html(code: str, height: int) -> None:
    """st.iframe on new Streamlit versions, components.html on older ones."""
    if hasattr(st, "iframe"):
        st.iframe(code, height=height)
    else:
        components.html(code, height=height)


def flip_card(concept: str, question: str, answer: str) -> None:
    """A real 3D flip card with heavy stationery luxury aesthetic. Click it to flip."""
    c, q, a = html.escape(concept), html.escape(question), html.escape(answer)
    show_html(
        f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;1,500&display=swap');
  body {{ margin: 0; background: transparent; font-family: 'Plus Jakarta Sans', -apple-system, sans-serif; }}
  .scene {{ perspective: 1200px; width: 100%; max-width: 580px; height: 280px; margin: 0 auto; animation: pop .4s cubic-bezier(0.16, 1, 0.3, 1); }}
  .card {{ position: relative; width: 100%; height: 100%; cursor: pointer; transition: transform .65s cubic-bezier(.4,.2,.2,1); transform-style: preserve-3d; }}
  .card.flipped {{ transform: rotateY(180deg); }}
  .face {{ position: absolute; inset: 0; backface-visibility: hidden; -webkit-backface-visibility: hidden; border-radius: 22px; padding: 28px 34px; box-sizing: border-box; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
  
  .front {{ 
    background: linear-gradient(145deg, #FFFFFF 0%, #FAF8F5 100%); 
    border: 1.5px solid #E6E1D6; 
    box-shadow: 0 16px 40px -10px rgba(50, 45, 35, 0.08), 0 2px 6px rgba(0,0,0,0.02); 
    color: #1A1A1E; 
  }}
  .front .text {{ font-family: 'Playfair Display', Georgia, serif; font-size: 23px; font-weight: 550; font-style: italic; line-height: 1.4; color: #18181B; max-height: 160px; overflow-y: auto; }}
  .front .tag {{ background: #F3EFE8; color: #8A6538; border: 1px solid #E5DFD4; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; font-weight: 700; border-radius: 999px; padding: 4px 12px; margin-bottom: 16px; }}
  .front .hint {{ position: absolute; bottom: 14px; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #9A958A; font-weight: 600; }}

  .back {{ 
    background: linear-gradient(145deg, #FCFDFC 0%, #F1F6F2 100%); 
    border: 1.5px solid #D6E4D8; 
    box-shadow: 0 16px 40px -10px rgba(35, 50, 40, 0.08), 0 2px 6px rgba(0,0,0,0.02); 
    color: #1A281F; 
    transform: rotateY(180deg); 
  }}
  .back .text {{ font-family: 'Plus Jakarta Sans', sans-serif; font-size: 19px; font-weight: 550; line-height: 1.45; color: #18281F; max-height: 160px; overflow-y: auto; }}
  .back .tag {{ background: #E5EFE6; color: #2E6B47; border: 1px solid #D0E2D3; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; font-weight: 700; border-radius: 999px; padding: 4px 12px; margin-bottom: 16px; }}
  .back .hint {{ position: absolute; bottom: 14px; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #728A7A; font-weight: 600; }}

  @keyframes pop {{ from {{ opacity: 0; transform: translateY(14px) scale(.98); }} to {{ opacity: 1; transform: none; }} }}
</style>
<div class="scene">
  <div class="card" onclick="this.classList.toggle('flipped')">
    <div class="face front">
      <div class="tag">{c} &middot; Question</div>
      <div class="text">{q}</div>
      <div class="hint">Click anywhere to reveal answer ↺</div>
    </div>
    <div class="face back">
      <div class="tag">Answer</div>
      <div class="text">{a}</div>
      <div class="hint">Click to flip back ↺</div>
    </div>
  </div>
</div>
""".strip(),
        height=310,
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
    one, two, three = st.columns(3)
    one.metric("Study decks", len(decks))
    two.metric("Cards saved", total_cards)
    three.metric("Cards reviewed", reviewed)
    st.markdown("<div class='deck-toolbar'><h3>Select Deck to Practice</h3><p>Choose a concept below and target all cards or focus specifically on missed items.</p></div>", unsafe_allow_html=True)
    st.dataframe(rows, hide_index=True, use_container_width=True)

    concept = st.selectbox("Pick a concept to study", list(decks))
    cards = decks[concept]
    missed = [c["id"] for c in cards if c["last"] == "wrong"]
    mode = st.radio("Which cards?", ["All cards", f"Only the ones I missed ({len(missed)})"], horizontal=True)
    only_missed = mode.startswith("Only")
    if only_missed and not missed:
        st.info("Nothing missed here, nice! Study all cards instead.")
    elif st.button("▶️ Start studying", type="primary"):
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

    st.button("← Back to decks", on_click=stop_study)

    if pos < total:
        card = cards[queue[pos]]
        st.progress(pos / total, text=f"{concept}: card {pos + 1} of {total}")
        flip_card(concept, card["question"], card["answer"])
        col1, col2 = st.columns(2)
        col1.button("✅ I got it", on_click=grade, args=(True, card["id"]))
        col2.button("❌ Missed it", on_click=grade, args=(False, card["id"]))
        return

    # finished: show the score
    results = study["results"]
    right = sum(results.values())
    st.subheader(f"🎉 Done with {concept}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Right", right)
    c2.metric("Missed", total - right)
    c3.metric("Score", f"{right / total:.0%}" if total else "-")
    if total and right == total:
        st.balloons()
    missed_ids = [i for i, ok in results.items() if not ok]
    if missed_ids:
        with st.expander(f"Review the {len(missed_ids)} you missed"):
            for i in missed_ids:
                st.markdown(f"**{cards[i]['question']}**  \n{cards[i]['answer']}")
        if st.button("🔁 Redo the ones I missed", type="primary"):
            start_study(concept, missed_ids)
            st.rerun()
    if st.button("🔄 Study the whole deck again"):
        start_study(concept, list(cards))
        st.rerun()


def flashcards_page():
    st.markdown("<div class='workspace-top'><div class='workspace-label'><span class='status-dot'></span> Active Recall Studio</div><div class='workspace-label'>Personal Study Decks</div></div>", unsafe_allow_html=True)
    st.markdown("<section class='hero'><div class='hero-kicker'>✦ Active Recall Library</div><h1>Build knowledge that endures.</h1><p>Master complex topics through high-retention 3D flashcards. Review full decks or focus specifically on cards you missed.</p></section>", unsafe_allow_html=True)
    decks = cards_store.load()
    if not decks:
        st.info("No flashcards yet. Go to the Chat page and ask: *make me flashcards about short-term memory*.")
        return
    if ss.study and ss.study["concept"] in decks:
        study_view(decks)
    else:
        ss.study = None
        overview(decks)


if page == "Chat":
    chat_page()
else:
    flashcards_page()
