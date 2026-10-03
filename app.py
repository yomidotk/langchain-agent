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
    """Apply the product UI without changing the app's interaction model."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
        :root { --ink:#19221e; --muted:#637069; --paper:#f8faf7; --line:#e3e9e3; --pine:#1d5b43; --pine-deep:#154532; --mint:#dff4e9; --cream:#fffdf8; }
        .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
        [data-testid="stHeader"] { background:rgba(248,250,247,.88); }
        [data-testid="stSidebar"] { background:#f0f5f0; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] > div:first-child { padding:1.35rem 1rem; }
        .block-container { max-width:1040px; padding-top:2.7rem; padding-bottom:3rem; }
        h1,h2,h3 { color:var(--ink) !important; letter-spacing:-.035em; }
        h1 { font-family:'Fraunces',Georgia,serif; font-size:clamp(2.1rem,4vw,3.2rem) !important; margin-bottom:.35rem !important; }
        h2 { font-family:'Fraunces',Georgia,serif; }
        .focusly-brand { display:flex; align-items:center; gap:.7rem; padding:.1rem .25rem 1.45rem; }
        .focusly-mark { display:grid; place-items:center; width:2.15rem; height:2.15rem; border-radius:.72rem; background:var(--pine); color:#fff; font-weight:700; font-size:1.2rem; box-shadow:0 7px 18px rgba(29,91,67,.18); }
        .focusly-brand strong { display:block; font-size:1.17rem; letter-spacing:-.04em; }
        .focusly-brand span { display:block; color:var(--muted); font-size:.72rem; margin-top:.12rem; }
        .page-kicker { color:var(--pine); font-size:.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.12em; margin-bottom:.35rem; }
        [data-testid="stChatMessage"] { background:var(--cream); border:1px solid var(--line); border-radius:1rem; padding:.85rem 1rem; margin-bottom:.8rem; box-shadow:0 2px 10px rgba(22,46,32,.025); }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background:var(--mint); border-color:#cae8d9; }
        [data-testid="stChatInput"] { border:1px solid #cddbd1; border-radius:1rem; background:#fff; box-shadow:0 8px 24px rgba(25,34,30,.07); }
        [data-testid="stChatInput"] textarea { font-family:'DM Sans',sans-serif; }
        .stButton > button { border-radius:.7rem; border:1px solid #d3ddd5; background:#fff; color:var(--ink); font-weight:600; transition:all .18s ease; }
        .stButton > button:hover { border-color:var(--pine); color:var(--pine); transform:translateY(-1px); }
        .stButton > button[kind="primary"] { background:var(--pine); border-color:var(--pine); color:#fff; }
        .stButton > button[kind="primary"]:hover { background:var(--pine-deep); color:#fff; }
        div[data-baseweb="select"] > div,.stTextInput input { border-radius:.7rem !important; border-color:#d5ded6 !important; background:#fff !important; }
        [data-testid="stRadio"] label { font-size:.92rem; }
        [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:.9rem; padding:.85rem 1rem; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:.9rem; overflow:hidden; }
        [data-testid="stExpander"] { border:1px solid var(--line); border-radius:.85rem; background:#fff; }
        .stAlert { border-radius:.8rem; }
        @media (max-width:700px) { .block-container { padding:1.5rem 1rem 2rem; } }
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
    st.markdown("<div class='focusly-brand'><div class='focusly-mark'>*</div><div><strong>Focusly</strong><span>Your personal study space</span></div></div>", unsafe_allow_html=True)
    page = st.radio("Page", ["Chat", "Flashcards"], label_visibility="collapsed")
    user_name = st.text_input("Your name", "Chiraz")
    level = st.selectbox("Your level", ["beginner", "intermediate", "advanced"], index=1)
    language = st.selectbox("Answer language", list(LANGUAGES))
    has_tts = bool(secret("ALIBABA_API_KEY"))
    voice, style = "Cherry", "Default voice"
    if has_tts:
        st.subheader("🔊 Voice")
        voice = st.selectbox("Voice", tts.VOICES)
        style = st.selectbox("Speaking style", list(tts.STYLES))
    st.button("+ New conversation", on_click=new_conversation, type="primary", use_container_width=True)
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
    st.markdown("<div class='page-kicker'>Your learning companion</div>", unsafe_allow_html=True)
    st.title("What are we exploring?")
    st.caption("Ask a question, share a topic, or attach an image to learn from.")
    for i, (role, text) in enumerate(history(values.get("messages", []))):
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

    if prompt:
        text = prompt.text or "Describe this image."
        to_agent = text
        chat_index.touch(thread_id, title=text)  # registers the chat in your list on its first message
        with st.chat_message("user"):
            st.markdown(text)
        if prompt.files:  # multimodal: a vision model describes the image, the agent gets the description
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
    st.dataframe(rows, hide_index=True)

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
    st.markdown("<div class='page-kicker'>Practice with purpose</div>", unsafe_allow_html=True)
    st.title("Flashcards")
    st.caption("Review your decks, track progress, and strengthen recall.")
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
