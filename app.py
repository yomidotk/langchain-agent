"""Streamlit UI for Study Buddy (chat + flip-card flashcards). Run with:  streamlit run app.py"""
# Browser localStorage is used to persist chat messages across page refreshes.

import asyncio
import html
import json as _json
import random
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

st.set_page_config(page_title="Study Buddy", page_icon="📚")

import cards_store  # noqa: E402
import session_store  # noqa: E402
from agent import HERE, Context, build_agent, describe_image, secret  # noqa: E402

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

    client = MultiServerMCPClient(
        {"study_tools": {"command": sys.executable, "args": [str(HERE / "mcp_server.py")], "transport": "stdio"}}
    )
    mcp_tools = run(client.get_tools())  # MCP
    agent = build_agent(mcp_tools, InMemorySaver())  # short-term memory
    return run, agent


run, agent = get_runtime()

# ---------------------------------------------------------------- session & state
# Keep thread_id in query params so page refresh (F5) stays in the exact same session
qp = st.query_params
thread_id = qp.get("session")
if not thread_id:
    thread_id = str(uuid.uuid4())
    qp["session"] = thread_id

ss = st.session_state
ss.setdefault("thread_id", thread_id)
if ss.thread_id != thread_id:
    ss.thread_id = thread_id

# Load saved session data (messages + notes) from session_store
saved = session_store.get_session(ss.thread_id)
if "messages" not in ss:
    ss.messages = list(saved.get("messages", []))
if "notes" not in ss:
    ss.notes = list(saved.get("notes", []))
ss.setdefault("pending", None)  # human-in-the-loop requests waiting for a decision
ss.setdefault("study", None)  # current flashcard session


def _sync_session():
    """Save current chat messages and notes to session_store."""
    session_store.save_session(ss.thread_id, ss.messages, ss.notes)


# -------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📚 Study Buddy")
    page = st.radio("Page", ["💬 Chat", "🃏 Flashcards"], label_visibility="collapsed")
    user_name = st.text_input("Your name", "Chiraz")
    level = st.selectbox("Your level", ["beginner", "intermediate", "advanced"], index=1)
    if st.button("New conversation"):
        new_id = str(uuid.uuid4())
        ss.thread_id = new_id
        ss.messages = []
        ss.notes = []
        ss.pending = None
        st.query_params["session"] = new_id
        session_store.save_session(new_id, [], [])
        st.rerun()

config = {"configurable": {"thread_id": ss.thread_id}}
context = Context(user_name=user_name, level=level)  # runtime context

# Sync notes between agent state and session_store
state = run(agent.aget_state(config))
agent_notes = (state.values.get("notes") or []) if state and state.values else []
if agent_notes:
    ss.notes = agent_notes
    _sync_session()
elif ss.notes:
    # State in InMemorySaver was lost (e.g. on page refresh or restart), restore it into agent state
    run(agent.aupdate_state(config, {"notes": ss.notes}))

with st.sidebar:
    st.subheader("📝 Notes (agent state)")
    for n in ss.notes:
        st.markdown(f"- {n}")
    if not ss.notes:
        st.caption("Ask me to save a note.")

    with st.expander("➕ Add note manually", expanded=False):
        new_note = st.text_input("Note", key="manual_note_input", label_visibility="collapsed", placeholder="Add note...")
        if st.button("Save", key="btn_save_note"):
            if new_note.strip():
                ss.notes = list(ss.notes) + [new_note.strip()]
                run(agent.aupdate_state(config, {"notes": ss.notes}))
                _sync_session()
                st.rerun()


# =================================================================== chat
def call_agent(payload):
    with st.spinner("Thinking..."):
        result = run(agent.ainvoke(payload, config=config, context=context))
    interrupts = result.get("__interrupt__")
    if interrupts:  # the agent paused: needs a human decision
        ss.pending = interrupts[0].value["action_requests"]
    else:
        ss.messages.append({"role": "assistant", "content": result["messages"][-1].text})
        # Check if the agent updated notes during tool execution
        state = run(agent.aget_state(config))
        if state and state.values and "notes" in state.values:
            ss.notes = state.values.get("notes") or []
        _sync_session()


def chat_page():
    st.title("💬 Chat")
    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if ss.pending:
        with st.chat_message("assistant"):
            st.warning("I need your approval before doing this:")
            for r in ss.pending:
                st.code(f"{r['name']}({r['args']})", language="python")
            col1, col2 = st.columns(2)
            decision = None
            if col1.button("✅ Approve"):
                decision = {"type": "approve"}
            if col2.button("❌ Reject"):
                decision = {"type": "reject", "message": "User said no."}
        if decision:
            decisions = [decision] * len(ss.pending)
            ss.pending = None
            call_agent(Command(resume={"decisions": decisions}))
            st.rerun()

    prompt = st.chat_input(
        "Ask me anything, or attach an image...",
        accept_file=True,
        file_type=["png", "jpg", "jpeg"],
        disabled=bool(ss.pending),
    )

    if prompt:
        text = prompt.text or "Describe this image."
        shown = text
        to_agent = text
        if prompt.files:  # multimodal: a vision model describes the image, the agent gets the description
            f = prompt.files[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                tmp.write(f.getvalue())
            with st.spinner("Looking at the image..."):
                description = run(describe_image(tmp.name, text))
            to_agent = f"{text}\n\n[Attached image, described by a vision model: {description}]"
            shown = f"{text}\n\n🖼️ *{f.name} attached*"
        ss.messages.append({"role": "user", "content": shown})
        _sync_session()
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
    st.title("🃏 Flashcards")
    decks = cards_store.load()
    if not decks:
        st.info("No flashcards yet. Go to the Chat page and ask: *make me flashcards about short-term memory*.")
        return
    if ss.study and ss.study["concept"] in decks:
        study_view(decks)
    else:
        ss.study = None
        overview(decks)


if page == "💬 Chat":
    chat_page()
else:
    flashcards_page()
