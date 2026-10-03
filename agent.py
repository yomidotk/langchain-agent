"""Study Buddy: one LangChain agent that uses everything from the course (except RAG, for now).

Course topic            -> where it shows up below
---------------------------------------------------------------
Create Agent            -> create_agent(...) in build_agent()
Foundational Models     -> make_model() + FAST / SMART models
Tools                   -> get_time, save_note, create_flashcards, send_email, ...
Short-Term Memory       -> InMemorySaver + thread_id
Multimodal Messages     -> image_message() + Alibaba vision model
MCP                     -> MultiServerMCPClient + mcp_server.py
Context and State       -> Context (runtime context) + StudyState (custom state)
Multi-Agent Systems     -> quiz-maker sub-agent wrapped as a tool
Middleware              -> personalize / ModelRouter / Summarization / HITL
Managing Long Convos    -> SummarizationMiddleware
Human-in-the-Loop       -> HumanInTheLoopMiddleware on send_email
Dynamic Agents          -> @dynamic_prompt + ModelRouter (dynamic model)
Agent Chat UI           -> graph.py + langgraph.json
"""

import asyncio
import base64
import mimetypes
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelRequest,
    SummarizationMiddleware,
    dynamic_prompt,
)
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing_extensions import NotRequired

import cards_store

load_dotenv()
HERE = Path(__file__).parent


# ---------------------------------------------------------------- Models
def secret(name: str, default: str = "") -> str:
    """Read a key from the environment (.env) or, if running inside Streamlit, from st.secrets."""
    if os.getenv(name):
        return os.environ[name]
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


DO_URL = "https://inference.do-ai.run/v1"
ALIBABA_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # Singapore / international


def do_model(name: str) -> ChatOpenAI:
    return ChatOpenAI(model=name, base_url=DO_URL, api_key=secret("DO_API_KEY"))


def alibaba_model(name: str) -> ChatOpenAI:
    return ChatOpenAI(model=name, base_url=ALIBABA_URL, api_key=secret("ALIBABA_API_KEY"))


# Both providers speak the OpenAI API, so ChatOpenAI works with a different base_url.
FAST = do_model(os.getenv("FAST_MODEL", "openai-gpt-oss-20b"))  # short chats
SMART = alibaba_model(os.getenv("SMART_MODEL", "qwen-plus")) if secret("ALIBABA_API_KEY") else FAST  # long chats
VISION = alibaba_model(os.getenv("VISION_MODEL", "qwen3-vl-plus"))  # gpt-oss can't see images


# ------------------------------------------------- Context and State
@dataclass
class Context:
    """Read-only info passed at invoke time (runtime context)."""

    user_name: str = "friend"
    level: str = "beginner"  # beginner | intermediate | advanced


class StudyState(AgentState):
    """Custom state that lives in the checkpoint, tools can update it."""

    notes: NotRequired[list[str]]


# ------------------------------------------------------------- Tools
@tool
def get_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%A %d %B %Y, %H:%M")


@tool
def get_profile(runtime: ToolRuntime[Context]) -> str:
    """Get the name and level of the person you are helping."""
    ctx = runtime.context
    return f"Name: {ctx.user_name}. Level: {ctx.level}."


@tool
def save_note(note: str, runtime: ToolRuntime) -> Command:
    """Save a short study note so it can be recalled later in the conversation."""
    notes = runtime.state.get("notes", [])
    return Command(
        update={
            "notes": notes + [note],
            "messages": [
                ToolMessage(f"Saved note #{len(notes) + 1}: {note}", tool_call_id=runtime.tool_call_id)
            ],
        }
    )


@tool
def list_notes(runtime: ToolRuntime) -> str:
    """List every note saved so far."""
    notes = runtime.state.get("notes", [])
    return "\n".join(f"{i + 1}. {n}" for i, n in enumerate(notes)) or "No notes yet."


class Card(BaseModel):
    question: str
    answer: str


@tool
def create_flashcards(concept: str, cards: list[Card]) -> str:
    """Create a deck of flashcards for a concept. Give 5-8 clear question/answer pairs.
    The student studies them on the Flashcards page of the app."""
    n = cards_store.add_cards(concept, [c.model_dump() for c in cards])
    return f"Created {n} flashcards for '{concept}'. Tell the student to open the Flashcards page to study them."


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. (Fake sender for the demo: it only prints.)
    A human must approve this before it runs (see HumanInTheLoopMiddleware)."""
    print(f"\n[EMAIL SENT] to={to} | subject={subject}\n{body}\n")
    return f"Email sent to {to}."


# ------------------------------------------------------- Multi-agent
def build_quiz_tool():
    """A sub-agent with its own prompt, exposed to the main agent as a tool."""
    quiz_agent = create_agent(
        model=FAST,
        tools=[],
        system_prompt=(
            "You write short quizzes. Given a topic, write 3 multiple-choice questions "
            "with 4 options each, and put the answers at the end."
        ),
    )

    @tool
    async def make_quiz(topic: str) -> str:
        """Delegate quiz creation on a topic to the quiz-maker sub-agent."""
        result = await quiz_agent.ainvoke({"messages": [{"role": "user", "content": topic}]})
        return result["messages"][-1].text

    return make_quiz


# ---------------------------------------------------------- Middleware
@dynamic_prompt
def personalize(request: ModelRequest) -> str:
    """Dynamic system prompt built from the runtime context."""
    ctx = request.runtime.context
    return (
        "You are Study Buddy, a friendly AI tutor.\n"
        f"The student is {ctx.user_name}, level: {ctx.level}. Adapt your explanations to that level.\n"
        "Use tools when they help: notes for things worth remembering, make_quiz for quizzes, "
        "create_flashcards when asked for flashcards (the student studies them on the Flashcards page). "
        "Never send an email without being asked."
    )


class ModelRouter(AgentMiddleware):
    """Dynamic model: FAST for short chats, SMART once the chat gets long."""

    def _route(self, request: ModelRequest) -> ModelRequest:
        return request.override(model=SMART if len(request.messages) > 12 else FAST)

    def wrap_model_call(self, request, handler):
        return handler(self._route(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._route(request))


# -------------------------------------------------------------- Agent
def build_agent(extra_tools=(), checkpointer=None):
    tools = [get_time, get_profile, save_note, list_notes, create_flashcards, send_email, build_quiz_tool(), *extra_tools]

    return create_agent(
        model=FAST,
        tools=tools,
        context_schema=Context,
        state_schema=StudyState,
        checkpointer=checkpointer,  # None when served by `langgraph dev` (it brings its own)
        middleware=[
            personalize,
            ModelRouter(),
            SummarizationMiddleware(model=FAST, trigger=("tokens", 4000), keep=("messages", 10)),
            HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),
        ],
    )


# --------------------------------------------------------- Multimodal
def image_message(path: str, question: str) -> HumanMessage:
    """Build a message with text + a local image."""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image", "base64": data, "mime_type": mime},
        ]
    )


async def describe_image(path: str, question: str) -> str:
    """gpt-oss can't see images, so this sends them to a vision model on Alibaba."""
    if not secret("ALIBABA_API_KEY"):
        return "No ALIBABA_API_KEY set, so I can't look at images yet."
    reply = await VISION.ainvoke([image_message(path, question)])
    return reply.text


# ----------------------------------------------------------- CLI chat
async def main():
    # MCP: connect to the local server and load its tools
    client = MultiServerMCPClient(
        {"study_tools": {"command": sys.executable, "args": [str(HERE / "mcp_server.py")], "transport": "stdio"}}
    )
    mcp_tools = await client.get_tools()

    agent = build_agent(mcp_tools, InMemorySaver())  # short-term memory
    config = {"configurable": {"thread_id": "session-1"}}
    context = Context(user_name="Chiraz", level="intermediate")

    print("Study Buddy ready. Type a message, '/image path.png question' for a picture, or 'quit'.")
    while True:
        text = input("\nyou> ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if text.startswith("/image "):
            _, path, *question = text.split(" ", 2)
            print(f"\nbuddy> {await describe_image(path, question[0] if question else 'Describe this image.')}")
            continue

        result = await agent.ainvoke({"messages": [{"role": "user", "content": text}]}, config=config, context=context)

        # Human-in-the-loop: the run pauses until we approve or reject
        while result.get("__interrupt__"):
            requests = result["__interrupt__"][0].value["action_requests"]
            decisions = []
            for r in requests:
                print(f"\nApproval needed: {r['name']}({r['args']})")
                ok = input("approve? [y/n] ").strip().lower() == "y"
                decisions.append({"type": "approve"} if ok else {"type": "reject", "message": "User said no."})
            result = await agent.ainvoke(Command(resume={"decisions": decisions}), config=config, context=context)

        print(f"\nbuddy> {result['messages'][-1].text}")


if __name__ == "__main__":
    asyncio.run(main())
