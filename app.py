"""Streamlit UI for Studeno (chat + flip-card flashcards + notes). Run with:  streamlit run app.py"""

import asyncio
import html
import logging
import random
import re
import sys
import threading
import uuid
from pathlib import Path

import aiosqlite
import streamlit as st
import streamlit.components.v1 as components
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

log = logging.getLogger(__name__)

# ── Free-tier message cap ──────────────────────────────────────────────────────
FREE_MSG_LIMIT = 10  # max user messages per chat thread before we pause the user

st.set_page_config(page_title="Studeno", page_icon="🦕", layout="centered")

import cards_store  # noqa: E402
import chat_index  # noqa: E402
import notes_store  # noqa: E402
import organizer  # noqa: E402
import tts  # noqa: E402
import userdata  # noqa: E402
from agent import HERE, QUIZ_ANSWERS_MARK, Context, build_agent, describe_image, describe_image_bytes, secret  # noqa: E402

st.markdown(
    """
<style>
  .block-container { padding-top: 2.2rem; max-width: 860px; }
  .hero { font-size: 2.1rem; font-weight: 800; line-height: 1.2; margin: 0 0 .2rem 0;
          background: linear-gradient(90deg, #6366f1, #10b981); -webkit-background-clip: text;
          background-clip: text; color: transparent; }
  .sub { opacity: .7; margin-bottom: 1.2rem; }
  [data-testid="stSidebar"] .stButton > button { width: 100%; justify-content: flex-start; border-radius: 10px; }
  [data-testid="stChatMessage"] { border-radius: 16px; padding: .8rem 1rem; }
  [class*="st-key-sug"] button { width: 100%; text-align: left; border-radius: 12px; padding: .8rem 1rem; }
  [data-testid="stMetric"] { background: rgba(99,102,241,.08); border-radius: 14px; padding: .8rem 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


def title(text: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="hero">{text}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sub">{subtitle}</div>', unsafe_allow_html=True)


if not secret("DO_API_KEY"):
    st.error("Add DO_API_KEY (and ALIBABA_API_KEY) to .streamlit/secrets.toml or a .env file.")
    st.stop()


@st.cache_resource
def get_runtime():
    """Build the agent once. Async calls run on one background event loop, because
    Streamlit reruns the script on every click and async HTTP clients dislike changing loops."""
    import sqlite3
    from langgraph.checkpoint.memory import InMemorySaver

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    def run(coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    async def make_saver():
        """Try SQLite in a writable temp dir; fall back to in-memory if the host blocks it."""
        import tempfile
        db_path = Path(tempfile.gettempdir()) / "studeno_checkpoints.db"
        try:
            conn = await aiosqlite.connect(str(db_path))
            # quick smoke-test: will raise if disk is locked / read-only
            await conn.execute("PRAGMA journal_mode=DELETE")
            await conn.commit()
            return AsyncSqliteSaver(conn)
        except Exception:
            return InMemorySaver()

    agent = build_agent(checkpointer=run(make_saver()))
    return run, agent


run, agent = get_runtime()

# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("study", None)  # current flashcard session
ss.setdefault("playing", None)  # which answer is being read aloud
PAGES = ["💬 Chat", "🃏 Flashcards", "📝 Notes"]
ss.setdefault("page", PAGES[0])

# Each visitor gets a private space, identified by a random key in the URL (?u=...).
# Nothing is shared between visitors: notes, flashcards and the chat list live in data/<key>/.
uid = st.query_params.get("u")
if not userdata.is_valid(uid) or uid == "local":
    uid = userdata.new_uid()
    st.query_params["u"] = uid


def load_state(chat_id: str):
    return run(agent.aget_state({"configurable": {"thread_id": chat_id}}))


# Which chat are we in? The URL (?chat=...) wins; otherwise reopen the chat this visitor used last.
index = chat_index.load(uid)
thread_id = st.query_params.get("chat") or index["last"] or chat_index.new_id()
state = load_state(thread_id)
if thread_id not in index["chats"] and (state.values or {}).get("messages"):
    thread_id = chat_index.new_id()  # that chat belongs to someone else, so start a fresh one
    state = load_state(thread_id)
st.query_params["chat"] = thread_id
chat_index.set_last(uid, thread_id)
config = {"configurable": {"thread_id": thread_id}}


def new_conversation():
    st.query_params["chat"] = chat_index.new_id()
    ss.page = PAGES[0]


def open_chat(chat_id: str):
    st.query_params["chat"] = chat_id
    ss.page = PAGES[0]


# -------------------------------------------------------------- sidebar
has_tts = bool(secret("ALIBABA_API_KEY"))
with st.sidebar:
    st.markdown(
        '<h1 style="text-align:center;font-size:2.3rem;font-weight:800;color:#6366f1;margin:0 0 1rem 0;letter-spacing:-.02em;">StuDeno</h1>',
        unsafe_allow_html=True,
    )
    st.segmented_control("Menu", PAGES, key="page", label_visibility="collapsed")
    st.button("➕ New chat", on_click=new_conversation, type="primary")

    past = chat_index.recent(uid, 8)
    if past:
        st.caption("RECENT CHATS")
        for cid, chat_title in past:
            st.button(f"💬 {chat_title}", key=f"chat_{cid}", on_click=open_chat, args=(cid,), disabled=cid == thread_id)

    st.divider()
    with st.expander("⚙️ Settings"):
        st.text_input("Your name", "student", key="name")
        st.selectbox("Your level", ["beginner", "intermediate", "advanced"], index=1, key="level")
        if has_tts:
            st.selectbox("Voice", tts.VOICES, key="voice")
            st.selectbox("Speaking style", list(tts.STYLES), key="style")
    st.caption("🔒 Your chats, notes and flashcards are private to this link. Bookmark this page to come back to them.")

page = ss.get("page") or PAGES[0]
values = state.values if state and state.values else {}  # everything saved for this chat: messages


def make_context() -> Context:  # runtime context, built fresh for every call so settings apply right away
    return Context(user_name=ss.get("name", "friend"), level=ss.get("level", "intermediate"), user_id=uid)


# =================================================================== chat
IMAGE_TAIL = re.compile(r"\n\n\[Attached image, described by a vision model:.*\]\s*$", re.DOTALL)
SUGGESTIONS = [
    "Explain RAG in simple words",
    "Make me flashcards about short-term memory",
    "Quiz me on middleware",
    "Save a note: middleware wraps the model call",
]


def history(messages):
    """Turn the saved agent messages into chat bubbles: (role, text, kind). Tool calls are hidden,
    except quizzes, whose text would otherwise never reach the screen."""
    out = []
    for m in messages:
        if m.additional_kwargs.get("lc_source") == "summarization":
            continue
        if m.type == "human":
            out.append(("user", IMAGE_TAIL.sub("\n\n🖼️ *image attached*", m.text), "chat"))
        elif m.type == "tool" and m.name == "make_quiz" and m.text.strip():
            out.append(("assistant", m.text, "quiz"))
        elif m.type == "ai" and m.text.strip():
            out.append(("assistant", m.text, "chat"))
    return out


def render_quiz(text: str):
    quiz, _, answers = text.partition(QUIZ_ANSWERS_MARK)
    st.markdown("📝 **Quiz time!**\n\n" + quiz.strip())
    if answers.strip():
        with st.expander("Show the answers"):
            st.markdown(answers.strip())


# ── Image validation helpers ───────────────────────────────────────────────────
_IMAGE_MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",   # JPEG
    b"\x89PNG": "image/png",          # PNG
}
_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB


def _validate_image(f) -> tuple[bytes, str] | None:
    """Return (raw_bytes, mime) if the file is a valid PNG/JPEG ≤5 MB, else None."""
    data = f.getvalue()
    if len(data) > _MAX_IMAGE_BYTES:
        st.error("Image must be under 5 MB.")
        return None
    for magic, mime in _IMAGE_MAGIC.items():
        if data.startswith(magic):
            return data, mime
    st.error("Only PNG and JPEG images are accepted.")
    return None


# ── Generic error helper ───────────────────────────────────────────────────────
def _server_error(exc: Exception, user_msg: str = "Something went wrong. Please try again."):
    """Show a safe generic message to the visitor; log the real error server-side."""
    log.exception("[studeno] %s", exc)
    st.error(user_msg)


def listen_ui(i: int, text: str):
    """A 🔊 Listen button under an answer. Click it to hear the answer instead of reading it."""
    key = f"{thread_id}_{i}"
    fresh = st.button("🔊 Listen", key=f"listen_{key}")
    if fresh:
        ss.playing = key
    if ss.playing == key:
        try:
            with st.spinner("Generating the voice..."):
                audio = tts.synthesize(text, secret("ALIBABA_API_KEY"), ss.get("voice", "Cherry"), ss.get("style"))
            st.audio(audio, format="audio/wav", autoplay=fresh)
        except Exception as exc:
            _server_error(exc, "Couldn't generate the voice right now.")


def call_agent(payload):
    try:
        with st.spinner("Thinking..."):
            run(agent.ainvoke(payload, config=config, context=make_context()))
    except Exception as exc:
        _server_error(exc)


def queue_prompt(text: str):
    ss.queued = text


def _count_user_messages(messages: list) -> int:
    """Count how many human messages are already in this chat thread."""
    return sum(1 for m in messages if getattr(m, "type", None) == "human")


def send(text: str, image_bytes: bytes | None = None, image_mime: str | None = None):
    chat_index.touch(uid, thread_id, title=text)
    with st.chat_message("user"):
        st.markdown(text)
    to_agent = text
    if image_bytes:  # kept entirely in memory, never touches disk
        with st.spinner("Looking at the image..."):
            try:
                description = run(describe_image_bytes(image_bytes, image_mime, text))
                to_agent = f"{text}\n\n[Attached image, described by a vision model: {description}]"
            except Exception as exc:
                _server_error(exc, "Couldn't analyse the image right now.")
                return
    call_agent({"messages": [{"role": "user", "content": to_agent}]})
    st.rerun()


def chat_page():
    title("Chat", "Ask anything, attach a screenshot, or turn what you learn into flashcards and quizzes.")
    queued = ss.pop("queued", None)
    raw_messages = values.get("messages", [])
    messages = history(raw_messages)

    # ── Free-tier cap ────────────────────────────────────────────────────────
    user_msg_count = _count_user_messages(raw_messages)
    limit_reached = user_msg_count >= FREE_MSG_LIMIT

    if not messages and not queued:
        st.markdown(f"#### Hi {ss.get('name', '')} 👋 What do you want to learn today?")
        cols = st.columns(2)
        for i, suggestion in enumerate(SUGGESTIONS):
            cols[i % 2].button(suggestion, key=f"sug{i}", on_click=queue_prompt, args=(suggestion,))

    for i, (role, text, kind) in enumerate(messages):
        with st.chat_message(role):
            if kind == "quiz":
                render_quiz(text)
                continue
            st.markdown(text)
            if role == "assistant" and has_tts:
                listen_ui(i, text)

    # Show remaining messages counter when getting close
    remaining = FREE_MSG_LIMIT - user_msg_count
    if 0 < remaining <= 3:
        st.caption(f"⚠️ {remaining} free message{'s' if remaining != 1 else ''} left in this chat.")

    if limit_reached:
        st.info(
            f"🎓 You've used all {FREE_MSG_LIMIT} free messages in this chat. "
            "Start a **New chat** from the sidebar to continue!",
            icon="🔒",
        )
        return  # no chat input rendered below — can't send more

    prompt = st.chat_input(
        "Ask me anything, or attach an image...",
        accept_file=True,
        file_type=["png", "jpg", "jpeg"],
    )
    if queued:
        send(queued)
    elif prompt:
        img_bytes, img_mime = None, None
        if prompt.files:
            result = _validate_image(prompt.files[0])
            if result is None:
                st.stop()  # error already shown by _validate_image
            img_bytes, img_mime = result
        send(prompt.text or "Describe this image.", img_bytes, img_mime)


# =================================================================== notes
def delete_note(note_id: str):
    notes_store.delete(uid, note_id)


def request_reorganize():
    ss.force_organize = True


def notes_page():
    title("Notes", "Your notes are sorted into notebooks by topic, automatically.")
    with st.form("add_note", clear_on_submit=True):
        text = st.text_input("Add a note yourself", placeholder="e.g. LoRA fine-tunes a model by training small adapter matrices")
        if st.form_submit_button("Add note") and text.strip():
            notes_store.add(uid, text.strip())
            st.rerun()

    notes = notes_store.load(uid)
    if not notes:
        st.info("No notes yet. In the chat, say something like *save a note: middleware wraps the model call*.")
        return

    # The organizer agent runs whenever there are new, unsorted notes (or when you ask for a fresh start).
    unsorted = tuple(n["id"] for n in notes if not n.get("topic"))
    forced = ss.pop("force_organize", False)
    if forced or (unsorted and ss.get("organize_tried") != unsorted):
        ss.organize_tried = unsorted
        done = False
        try:
            with st.spinner("Organizing your notes into notebooks..."):
                run(organizer.organize_notes(uid, fresh=forced))
            done = True
        except Exception as e:
            st.warning(f"Couldn't organize the notes right now ({e}). They're shown unsorted for now.")
        if done:
            st.rerun()

    st.button("🔄 Re-organize from scratch", on_click=request_reorganize, help="Let the organizer regroup all notes")

    notebooks: dict[str, list[dict]] = {}
    for n in notes:
        notebooks.setdefault(n.get("topic") or "📥 Unsorted", []).append(n)
    for topic, items in sorted(notebooks.items(), key=lambda kv: (kv[0].startswith("📥"), kv[0].lower())):
        label = topic if topic.startswith("📥") else f"📓 {topic}"
        with st.expander(f"{label}  ·  {len(items)}", expanded=True):
            for n in items:
                c1, c2 = st.columns([12, 1])
                c1.markdown(f"- {n['text']}")
                c2.button("🗑️", key=f"del_{n['id']}", on_click=delete_note, args=(n["id"],), help="Delete this note")


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
    cards_store.record(uid, study["concept"], card_id, correct)
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
    total = sum(r["Cards"] for r in rows)
    m1, m2, m3 = st.columns(3)
    m1.metric("Cards", total)
    m2.metric("Got it ✅", sum(r["Got it ✅"] for r in rows))
    m3.metric("To redo ❌", sum(r["To redo ❌"] for r in rows))
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
            cards_store.delete_deck(uid, concept)
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
    title("Flashcards", "Click a card to flip it, then say whether you got it. Missed cards can be redone later.")
    decks = cards_store.load(uid)
    if not decks:
        st.info("No flashcards yet. Go to the Chat page and ask: *make me flashcards about short-term memory*.")
        return
    if ss.study and ss.study["concept"] in decks:
        study_view(decks)
    else:
        ss.study = None
        overview(decks)


if page == PAGES[0]:
    chat_page()
elif page == PAGES[1]:
    flashcards_page()
else:
    notes_page()
