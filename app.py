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
    """Apply the Focusly product interface without changing app behavior."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
        :root { --navy:#101828; --navy-2:#18243a; --ink:#172033; --muted:#697386; --canvas:#f5f7fb; --card:#ffffff; --line:#e5e9f0; --blue:#4f46e5; --violet:#7c3aed; --blue-pale:#eef2ff; --success:#16803c; }
        .stApp { background:var(--canvas); color:var(--ink); font-family:'DM Sans',sans-serif; }
        [data-testid="stHeader"] { background:rgba(245,247,251,.86); border-bottom:1px solid rgba(229,233,240,.7); }
        [data-testid="stSidebar"] { background:var(--navy); border:0; }
        [data-testid="stSidebar"] > div:first-child { padding:1.25rem .95rem 1.6rem; }
        [data-testid="stSidebar"] * { color:#e8edf8; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#97a5c1 !important; }
        [data-testid="stSidebar"] .stTextInput label, [data-testid="stSidebar"] .stSelectbox label { color:#aebad0 !important; font-size:.73rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div, [data-testid="stSidebar"] .stTextInput input { background:var(--navy-2) !important; border-color:#30415f !important; color:#fff !important; }
        [data-testid="stSidebar"] .stButton > button { background:transparent; border-color:transparent; color:#cdd7eb; text-align:left; padding:.58rem .65rem; }
        [data-testid="stSidebar"] .stButton > button:hover { background:#223252; border-color:#304566; color:#fff; transform:none; }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] { background:linear-gradient(110deg,var(--blue),var(--violet)); border:0; color:#fff; text-align:center; padding:.68rem .7rem; box-shadow:0 10px 24px rgba(79,70,229,.28); }
        [data-testid="stSidebar"] [data-testid="stRadio"] { gap:.35rem; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { width:100%; padding:.62rem .65rem; border-radius:.6rem; transition:.15s ease; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:#263858; color:#fff; font-weight:700; }
        [data-testid="stSidebar"] [data-testid="stRadio"] input { accent-color:#818cf8; }
        .block-container { max-width:1180px; padding:2.2rem 2.6rem 3rem; }
        h1,h2,h3 { color:var(--ink) !important; letter-spacing:-.035em; }
        h1 { font-family:'Playfair Display',Georgia,serif; font-size:clamp(2.1rem,4vw,3.25rem) !important; line-height:1.08; margin:0 !important; }
        h2 { font-size:1.32rem !important; margin-top:1.8rem; }
        .focusly-brand { display:flex; align-items:center; gap:.72rem; padding:.15rem .3rem 1.7rem; }
        .focusly-mark { display:grid; place-items:center; width:2.25rem; height:2.25rem; border-radius:.72rem; background:linear-gradient(140deg,#8b5cf6,#4f46e5); color:#fff; font-weight:700; font-size:1.1rem; box-shadow:0 8px 20px rgba(79,70,229,.35); }
        .focusly-brand strong { display:block; color:#fff; font-size:1.2rem; letter-spacing:-.04em; }
        .focusly-brand span { display:block; color:#9eacc8; font-size:.7rem; margin-top:.1rem; }
        .sidebar-section { color:#8d9bb7; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.13em; padding:.9rem .55rem .35rem; }
        .workspace-top { display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:1.8rem; }
        .workspace-label { display:flex; align-items:center; gap:.55rem; color:var(--muted); font-size:.84rem; font-weight:600; }
        .status-dot { width:.48rem; height:.48rem; border-radius:50%; background:#20b15a; box-shadow:0 0 0 4px #e2f7e9; }
        .hero { position:relative; overflow:hidden; padding:2.25rem 2.4rem; border-radius:1.35rem; color:#fff; background:linear-gradient(118deg,#18254a 0%,#263c73 55%,#594bb7 100%); box-shadow:0 18px 44px rgba(30,46,89,.18); margin-bottom:1.5rem; }
        .hero:after { content:''; position:absolute; width:20rem; height:20rem; border:1px solid rgba(255,255,255,.13); border-radius:50%; right:-6rem; top:-10rem; box-shadow:-4rem 5rem 0 -1px rgba(255,255,255,.06), -8rem 9rem 0 -1px rgba(255,255,255,.05); }
        .hero > * { position:relative; z-index:1; }
        .hero-kicker { font-size:.73rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#bfc9ff; margin-bottom:.6rem; }
        .hero h1 { color:#fff !important; max-width:40rem; }
        .hero p { color:#d6def5; max-width:36rem; margin:.8rem 0 0; font-size:1rem; line-height:1.55; }
        .starter-heading { font-size:.78rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase; color:var(--muted); margin:1.6rem 0 .65rem; }
        .stButton > button { min-height:2.55rem; border-radius:.7rem; border:1px solid #d9dfeb; background:var(--card); color:var(--ink); font-weight:650; transition:all .16s ease; }
        .stButton > button:hover { border-color:var(--blue); color:var(--blue); transform:translateY(-1px); box-shadow:0 7px 17px rgba(68,76,111,.1); }
        .starter-card .stButton > button { height:6.3rem; white-space:normal; text-align:left; padding:1rem; align-items:flex-start; background:#fff; }
        .stButton > button[kind="primary"] { background:linear-gradient(110deg,var(--blue),var(--violet)); border:0; color:#fff; }
        .stButton > button[kind="primary"]:hover { color:#fff; box-shadow:0 10px 22px rgba(79,70,229,.25); }
        [data-testid="stChatMessage"] { background:#fff; border:1px solid var(--line); border-radius:1rem; padding:1rem 1.05rem; margin-bottom:.85rem; box-shadow:0 3px 14px rgba(23,32,51,.035); }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background:var(--blue-pale); border-color:#dce3ff; }
        [data-testid="stChatInput"] { border:1px solid #d7deec; border-radius:1rem; background:#fff; box-shadow:0 10px 26px rgba(27,42,75,.08); }
        [data-testid="stChatInput"] textarea { font-family:'DM Sans',sans-serif; }
        div[data-baseweb="select"] > div,.stTextInput input { border-radius:.65rem !important; border-color:#d9dfeb !important; background:#fff !important; }
        [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:1rem; padding:1rem 1.1rem; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:1rem; overflow:hidden; background:#fff; }
        [data-testid="stExpander"] { border:1px solid var(--line); border-radius:.9rem; background:#fff; }
        .deck-toolbar { background:#fff; border:1px solid var(--line); border-radius:1rem; padding:1.25rem; margin:1rem 0 1.25rem; }
        .deck-toolbar h3 { margin:0 0 .25rem; font-size:1.1rem; }
        .deck-toolbar p { color:var(--muted); margin:0; font-size:.9rem; }
        .stAlert { border-radius:.85rem; }
        @media (max-width:700px) { .block-container { padding:1.35rem 1rem 2.25rem; } .hero { padding:1.65rem; } .workspace-top { margin-bottom:1.25rem; } }
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
    st.markdown("<div class='focusly-brand'><div class='focusly-mark'>F</div><div><strong>Focusly</strong><span>Learn with clarity</span></div></div>", unsafe_allow_html=True)
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
    st.subheader("📝 Notes")
    saved_notes = notes_store.load()
    for n in saved_notes:
        st.markdown(f"- {n}")
    if not saved_notes:
        st.caption("Ask me to save a note.")


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
    st.markdown("<div class='workspace-top'><div class='workspace-label'><span class='status-dot'></span> Focus session active</div><div class='workspace-label'>AI study workspace</div></div>", unsafe_allow_html=True)
    quick_prompt = None
    if not messages:
        safe_name = html.escape(user_name or "there")
        st.markdown(f"<section class='hero'><div class='hero-kicker'>Your learning space</div><h1>Make your next study session count, {safe_name}.</h1><p>Ask for a clear explanation, turn a topic into flashcards, or bring in an image you want to understand.</p></section>", unsafe_allow_html=True)
        st.markdown("<div class='starter-heading'>Start with a guided prompt</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='starter-card'>", unsafe_allow_html=True)
            if st.button("Explain a difficult topic simply", key="starter_explain", use_container_width=True):
                quick_prompt = "Explain a difficult topic to me in simple terms."
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='starter-card'>", unsafe_allow_html=True)
            if st.button("Create a focused revision plan", key="starter_plan", use_container_width=True):
                quick_prompt = "Create a focused revision plan for me."
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='starter-card'>", unsafe_allow_html=True)
            if st.button("Turn a topic into flashcards", key="starter_cards", use_container_width=True):
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
            st.warning("I need your approval before doing this:")
            for r in pending:
                st.code(f"{r['name']}({r['args']})", language="python")
            col1, col2 = st.columns(2)
            decision = None
            if col1.button("✅ Approve"):
                decision = {"type": "approve"}
            if col2.button("❌ Reject"):
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
    """A real 3D flip card (pure CSS/JS in an iframe). Click it to flip."""
    c, q, a = html.escape(concept), html.escape(question), html.escape(answer)
    show_html(
        f"""
<style>
  body {{ margin: 0; font-family: system-ui, -apple-system, sans-serif; }}
  .scene {{ perspective: 1100px; width: 100%; max-width: 580px; height: 270px; margin: 0 auto;
            animation: pop .45s ease-out; }}
  .card {{ position: relative; width: 100%; height: 100%; cursor: pointer;
           transition: transform .7s cubic-bezier(.4,.2,.2,1); transform-style: preserve-3d; }}
  .card.flipped {{ transform: rotateY(180deg); }}
  .face {{ position: absolute; inset: 0; backface-visibility: hidden; -webkit-backface-visibility: hidden;
           border-radius: 18px; padding: 24px; box-sizing: border-box; color: white; text-align: center;
           display: flex; flex-direction: column; align-items: center; justify-content: center;
           box-shadow: 0 10px 30px rgba(0,0,0,.28); }}
  .front {{ background: linear-gradient(135deg, #6366f1, #8b5cf6); }}
  .back  {{ background: linear-gradient(135deg, #0ea5e9, #10b981); transform: rotateY(180deg); }}
  .tag   {{ font-size: 12px; letter-spacing: .12em; text-transform: uppercase; opacity: .8; margin-bottom: 12px; }}
  .text  {{ font-size: 22px; line-height: 1.35; font-weight: 600; max-height: 170px; overflow-y: auto; }}
  .hint  {{ position: absolute; bottom: 12px; font-size: 12px; opacity: .7; }}
  @keyframes pop {{ from {{ opacity: 0; transform: translateY(16px) scale(.96); }}
                    to   {{ opacity: 1; transform: none; }} }}
</style>
<div class="scene">
  <div class="card" onclick="this.classList.toggle('flipped')">
    <div class="face front"><div class="tag">{c} &middot; question</div><div class="text">{q}</div>
      <div class="hint">click the card to flip</div></div>
    <div class="face back"><div class="tag">answer</div><div class="text">{a}</div>
      <div class="hint">click to flip back</div></div>
  </div>
</div>
""".strip(),
        height=300,
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
    st.markdown("<div class='deck-toolbar'><h3>Choose your next review</h3><p>Pick a deck and set the practice mode that works for this session.</p></div>", unsafe_allow_html=True)
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
    st.markdown("<div class='workspace-top'><div class='workspace-label'><span class='status-dot'></span> Recall practice</div><div class='workspace-label'>Your study library</div></div>", unsafe_allow_html=True)
    st.markdown("<section class='hero'><div class='hero-kicker'>Flashcard library</div><h1>Build knowledge that sticks.</h1><p>Use short, focused practice sessions and revisit the cards that need another pass.</p></section>", unsafe_allow_html=True)
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
