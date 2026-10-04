# 🦕 Studeno — AI-Powered Study Assistant

> A full-stack, multi-agent AI study buddy built with **LangChain**, **LangGraph**, **Streamlit**, and the **Model Context Protocol (MCP)**. Chat with an AI tutor, generate flashcards, take quizzes, save notes, and listen to answers read aloud — all in one app.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Chat** | Conversational AI tutor with short-term memory across sessions |
| 🃏 **Flashcards** | Auto-generated flip-card decks, with progress tracking and spaced-repetition retries |
| 📝 **Notes** | Save study notes from chat; auto-organized into topic notebooks by an AI organizer agent |
| 🔊 **Text-to-Speech** | Listen to any AI answer via Alibaba Qwen-TTS with selectable voices and styles |
| 🖼️ **Image Understanding** | Attach PNG/JPEG screenshots; a vision model describes them before passing to the tutor |
| 🧩 **Quizzes** | A dedicated quiz sub-agent writes 3-question multiple-choice quizzes on demand |
| 🔒 **Human-in-the-Loop** | The agent pauses and asks for your approval before sending an email |
| 🧠 **Dynamic Model Routing** | Switches between a fast model (short chats) and a smart model (long chats) automatically |
| 📝 **Conversation Summarization** | Long conversations are automatically summarized to stay within token limits |
| 🔑 **Private Workspaces** | Each visitor gets a unique URL key; data is fully isolated per user |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                   │
│   Chat Page │ Flashcards Page │ Notes Page                  │
└──────────────────────┬──────────────────────────────────────┘
                       │  invoke / state
┌──────────────────────▼──────────────────────────────────────┐
│                   Main Agent (agent.py)                     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Middleware Stack                    │   │
│  │  personalize → ModelRouter → Summarization → HITL   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Tools: get_time │ get_profile │ save_note │ list_notes │   │
│         create_flashcards │ send_email │ make_quiz (sub) │   │
│                          + MCP tools (word_count, etc.)  │   │
└────┬─────────────────────────────┬────────────────────────┘
     │ checkpointer (SQLite/memory) │ organizer agent
     │                             ▼
┌────▼───────────┐     ┌────────────────────────┐
│  LangGraph     │     │  Organizer (organizer. │
│  Checkpoint    │     │  py) sorts notes into  │
│  SQLite DB     │     │  topic notebooks       │
└────────────────┘     └────────────────────────┘
     │
     ├── cards_store.py  → data/<uid>/flashcards.json
     ├── notes_store.py  → data/<uid>/notes.json
     ├── chat_index.py   → data/<uid>/chats.json
     └── userdata.py     → data/<uid>/  (private user folder)
```

---

## 📁 File Reference

| File | Purpose |
|---|---|
| `app.py` | Streamlit entry point — all UI pages (Chat, Flashcards, Notes), session wiring, image upload |
| `agent.py` | Core agent: tools, middleware, model setup, multimodal helpers, CLI runner |
| `organizer.py` | Second AI agent that groups notes into topic notebooks automatically |
| `mcp_server.py` | Tiny MCP server (word counter + flashcard formatter) launched as a stdio subprocess |
| `cards_store.py` | Read/write flashcard decks and per-card progress to JSON |
| `notes_store.py` | Read/write study notes and their topic labels to JSON |
| `chat_index.py` | Track which chat threads belong to a visitor, and their titles |
| `userdata.py` | Validate/create user IDs and their private `data/<uid>/` directories |
| `tts.py` | Text-to-Speech via Alibaba Qwen-TTS: clean markdown, chunk text, cache WAV files |
| `graph.py` | Minimal entry point for `langgraph dev` / the Agent Chat UI |
| `requirements.txt` | Python dependencies |

---

## 🤖 Agent Deep Dive (`agent.py`)

### Models

The app uses **two AI providers**, both accessed via the OpenAI-compatible API:

| Constant | Provider | Default Model | When Used |
|---|---|---|---|
| `FAST` | DigitalOcean AI | `openai-gpt-oss-20b` | Short chats, quiz sub-agent, organizer |
| `SMART` | Alibaba Cloud | `qwen-plus` | Long chats (>12 messages) |
| `VISION` | Alibaba Cloud | `qwen3-vl-plus` | Image understanding |

### Tools

| Tool | Description |
|---|---|
| `get_time` | Returns current date & time |
| `get_profile` | Returns the student's name and level from runtime context |
| `save_note` | Persists a note to the user's notes file and updates chat state |
| `list_notes` | Lists all notes with their topic/notebook labels |
| `create_flashcards` | Generates a named deck of Q&A flashcard pairs |
| `send_email` | Fake email sender (requires human approval before running) |
| `make_quiz` | Invokes the quiz sub-agent and returns formatted questions + answers |
| MCP tools | `word_count`, `make_flashcard` (from the local MCP server) |

### Middleware Stack

Applied in order on every model call:

```
personalize              → injects a dynamic system prompt with user name & level
ModelRouter              → swaps FAST → SMART when conversation exceeds 12 messages
SummarizationMiddleware  → summarizes old messages when token count > 4000, keeps last 10
HumanInTheLoopMiddleware → pauses execution and waits for user approval on send_email
```

### Context & State

- **`Context`** (runtime) — passed at invoke time: `user_name`, `level`, `user_id`
- **`StudyState`** (checkpoint) — extends `AgentState` with a `notes` list that tools update via `Command(update=...)`

### Multi-Agent System

`make_quiz` is a **tool-wrapped sub-agent**: the main agent calls it like a tool, but internally it runs a completely separate `create_agent` with its own system prompt that generates 3-question multiple-choice quizzes with an answer key.

The **organizer** (`organizer.py`) is a second standalone agent invoked from the Notes page to group all notes into topic notebooks and respond with structured JSON.

---

## 🖼️ Multimodal Image Flow

1. User uploads a PNG/JPEG (≤ 5 MB) in the chat input.
2. `_validate_image()` checks magic bytes (not just file extension) and size.
3. Image bytes are sent to `describe_image_bytes()` which calls the **Alibaba Vision model** (`qwen3-vl-plus`).
4. The description is appended to the user's message as `[Attached image, described by a vision model: ...]`.
5. The enriched text message is sent to the main agent — **no image ever touches disk**.

---

## 🔒 Privacy & Session Model

Each visitor is identified by a **32-character random hex ID** stored in the URL as `?u=<uid>`. This ID:

- Is validated with a strict regex to prevent path traversal attacks (`^[a-f0-9]{32}$`).
- Maps to a private folder `data/<uid>/` containing `notes.json`, `flashcards.json`, and `chats.json`.
- Is generated fresh for every new visitor (or if an invalid/missing ID is found in the URL).

**Bookmark your URL** to return to your data — there is no login system.

The free-tier chat cap is **10 user messages per chat thread**. Start a new chat to continue.

---

## 🃏 Flashcard System

1. Ask the agent: *"Make me flashcards about neural networks"*
2. Agent calls `create_flashcards(concept, cards)` → saved to `data/<uid>/flashcards.json`
3. On the **Flashcards** page, pick a deck and click ▶️ Start studying
4. A **3D CSS flip card** appears — click to reveal the answer (pure CSS/JS, no framework)
5. Mark ✅ **Got it** or ❌ **Missed it** — right/wrong counts and last result are tracked per card
6. At the end, see your score and optionally **redo only the missed cards**
7. Cards are shuffled at the start of each session

---

## 🔊 Text-to-Speech (`tts.py`)

- Uses **Alibaba Qwen-TTS** (Singapore/international endpoint).
- Cleans markdown, links, code blocks, emojis, and bullet symbols before speaking.
- Splits text into ≤ 450-character sentence-aligned chunks (API limit ≈ 600 chars).
- Joins multiple WAV chunks into a single audio stream using Python's `wave` module.
- **Caches** generated audio to `audio_cache/` by MD5 hash of content — replays are instant and free.

**Voices:** Cherry, Serena, Ethan, Chelsie, Momo, Vivian

**Speaking Styles:**
| Style | Instruction |
|---|---|
| Default voice | No instruction (standard TTS model) |
| Friendly teacher | Warm, upbeat and encouraging |
| Calm and clear | Calm, soft, slightly slower pace |
| Energetic coach | Enthusiastic and motivating |

---

## 🔧 Setup & Running

### 1. Prerequisites

- Python 3.11+
- A **DigitalOcean AI** API key (`DO_API_KEY`) — **required**
- An **Alibaba Cloud** API key (`ALIBABA_API_KEY`) — optional (enables SMART model, Vision, and TTS)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure secrets

Create `.streamlit/secrets.toml`:

```toml
DO_API_KEY = "your-digitalocean-api-key"
ALIBABA_API_KEY = "your-alibaba-api-key"   # optional
```

Or use a `.env` file in the project root:

```env
DO_API_KEY=your-digitalocean-api-key
ALIBABA_API_KEY=your-alibaba-api-key
```

You can override the default model names via environment variables:

```env
FAST_MODEL=openai-gpt-oss-20b
SMART_MODEL=qwen-plus
VISION_MODEL=qwen3-vl-plus
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Your private URL will be `http://localhost:8501/?u=<your-uid>`.

### 5. Run in CLI mode (no UI)

```bash
python agent.py
```

Type messages directly at the prompt. Use `/image path.png What is this?` to send an image.

### 6. Run with LangGraph Dev UI

```bash
langgraph dev
```

Uses `graph.py` as the entry point and `langgraph.json` for configuration. The server provides its own checkpointing, so no SQLite is needed.

---

## 🌐 MCP Server (`mcp_server.py`)

The app spawns a local **Model Context Protocol** server as a subprocess over `stdio`. It exposes two utility tools to the agent:

| Tool | Description |
|---|---|
| `word_count(text)` | Counts the number of words in a string |
| `make_flashcard(question, answer)` | Formats a Q/A pair as `Q: ...\nA: ...` |

The server is built with `FastMCP` and is launched automatically by both the Streamlit app and the CLI runner via `MultiServerMCPClient`.

---

## 💡 Example Chat Prompts

```
"Explain transformer attention in simple words"
"Make me flashcards about the OSI model"
"Quiz me on Python decorators"
"Save a note: LoRA fine-tunes a model with small adapter matrices"
"What does this screenshot show?"  ← attach an image
"Send an email to alice@example.com summarizing today's session"
"How many words are in the following text: ..."
```

---

## 📦 Dependencies

```
streamlit>=1.43
langchain>=1.0
langgraph>=1.0
langchain-openai
langchain-mcp-adapters
mcp
langgraph-checkpoint-sqlite
aiosqlite
httpx
python-dotenv
langgraph-cli[inmem]
```

---

## 🎓 LangChain Course Concepts Covered

This project is a practical demonstration of the following course topics:

| Concept | Where in the code |
|---|---|
| Create Agent | `build_agent()` in `agent.py` |
| Foundational Models | `do_model()` / `alibaba_model()` + FAST / SMART / VISION |
| Tools | `get_time`, `save_note`, `create_flashcards`, `send_email`, ... |
| Short-Term Memory | `checkpointer` + `thread_id` in LangGraph |
| Multimodal Messages | `image_message()` + Alibaba vision model |
| MCP | `MultiServerMCPClient` + `mcp_server.py` |
| Context and State | `Context` (runtime) + `StudyState` (custom state schema) |
| Multi-Agent Systems | Quiz-maker sub-agent wrapped as a `make_quiz` tool |
| Middleware | `personalize` / `ModelRouter` / `SummarizationMiddleware` / HITL |
| Managing Long Convos | `SummarizationMiddleware` |
| Human-in-the-Loop | `HumanInTheLoopMiddleware` on `send_email` |
| Dynamic Agents | Dynamic system prompt (name, level) + `ModelRouter` |
| Agent Chat UI | `graph.py` + `langgraph.json` |

---

## 📜 License

For educational and demonstration purposes.
