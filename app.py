"""Streamlit chat UI for Study Buddy. Run with:  streamlit run app.py"""

import asyncio
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import streamlit as st
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

st.set_page_config(page_title="Study Buddy", page_icon="📚")

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

# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("thread_id", str(uuid.uuid4()))
ss.setdefault("messages", [])  # what we display: {"role", "content"}
ss.setdefault("pending", None)  # human-in-the-loop requests waiting for a decision

# -------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📚 Study Buddy")
    user_name = st.text_input("Your name", "Chiraz")
    level = st.selectbox("Your level", ["beginner", "intermediate", "advanced"], index=1)
    if st.button("New conversation"):
        ss.thread_id, ss.messages, ss.pending = str(uuid.uuid4()), [], None
        st.rerun()

config = {"configurable": {"thread_id": ss.thread_id}}
context = Context(user_name=user_name, level=level)  # runtime context

with st.sidebar:
    st.subheader("📝 Notes (agent state)")
    state = run(agent.aget_state(config))
    notes = state.values.get("notes", []) if state and state.values else []
    for n in notes:
        st.markdown(f"- {n}")
    if not notes:
        st.caption("Ask me to save a note.")


def call_agent(payload):
    with st.spinner("Thinking..."):
        result = run(agent.ainvoke(payload, config=config, context=context))
    interrupts = result.get("__interrupt__")
    if interrupts:  # the agent paused: needs a human decision
        ss.pending = interrupts[0].value["action_requests"]
    else:
        ss.messages.append({"role": "assistant", "content": result["messages"][-1].text})


# ----------------------------------------------------------------- chat
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
    call_agent({"messages": [{"role": "user", "content": to_agent}]})
    st.rerun()
