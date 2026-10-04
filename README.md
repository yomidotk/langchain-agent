# 🦕 Studeno — AI-Powered Study Assistant

> A full-stack, multi-agent AI study buddy built with **LangChain**, **LangGraph**, and **Streamlit**. Chat with an AI tutor, generate interactive flashcards, take quizzes, organize study notes, and listen to answers read aloud.

---

## ✨ Functional Features

| Feature | Description |
|---|---|
| 💬 **Conversational AI Tutor** | Intelligent tutor with short-term memory across sessions, adapting explanations to your skill level |
| 🃏 **Interactive 3D Flashcards** | AI-generated question/answer decks with interactive 3D flip animations, progress tracking, and missed-card retries |
| 📝 **Smart Notes & Notebooks** | Capture notes from chat or manually; an automated organizer agent sorts them into topic notebooks |
| 🧩 **Quiz Sub-Agent** | Generates 3-question multiple-choice quizzes on any topic with hidden expandable answer keys |
| 🔊 **Voice Text-to-Speech** | Listen to any tutor response via Alibaba Qwen-TTS with audio caching, selectable voices, and speaking styles |
| 🖼️ **Multimodal Vision** | Upload PNG/JPEG screenshots and notes; vision models describe diagrams and questions in memory |
| 🧠 **Dynamic Model Routing** | Automatically routes short chats to a fast model and switches to a high-capacity model as conversations grow |
| 📜 **Conversation Summarization** | Token-aware middleware automatically summarizes older context when exceeding token limits |
| 🔒 **Private Workspaces** | Isolated per-user storage keyed by a unique URL parameter with path-traversal protection |
| ⏱️ **Usage Guardrails** | Configurable message cap per chat session with reminder notices and seamless thread restarts |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                   │
│         💬 Chat  │  🃏 Flashcards  │  📝 Notes              │
└──────────────────────┬──────────────────────────────────────┘
                       │ invoke / state
┌──────────────────────▼──────────────────────────────────────┐
│                   Main Agent (agent.py)                     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Middleware Stack                    │   │
│  │  personalize (dynamic prompt) →                      │   │
│  │  ModelRouter (FAST ➔ SMART) →                       │   │
│  │  SummarizationMiddleware (token-triggered)           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Tools: get_time │ get_profile │ save_note │ list_notes │   │
│         create_flashcards │ make_quiz (sub-agent tool)      │
└────┬─────────────────────────────┬──────────────────────────┘
     │ checkpointer (SQLite/Memory)│ secondary agent
     │                             ▼
┌────▼───────────┐     ┌────────────────────────┐
│  LangGraph     │     │  Organizer Agent       │
│  Checkpoint    │     │  (organizer.py) sorts  │
│  SQLite DB     │     │  notes into notebooks  │
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
| `app.py` | Streamlit entry point — handles all UI views (Chat, Flashcards, Notes), session state, and file uploads |
| `agent.py` | Core agent definition — tools, middleware stack, model routing, multimodal vision, and CLI runner |
| `organizer.py` | Dedicated secondary agent that categorizes student notes into topic notebooks via structured JSON |
| `cards_store.py` | Thread-safe storage for flashcard decks, scores, and review statuses |
| `notes_store.py` | Thread-safe storage for student notes and notebook topic assignments |
| `chat_index.py` | Tracks chat history, timestamps, and thread metadata per visitor |
| `userdata.py` | Secure workspace isolation and user ID validation (`data/<uid>/`) |
| `tts.py` | Text-to-Speech engine — cleans text, chunks requests, joins WAVs, and caches audio to disk |
| `graph.py` | Entry point for `langgraph dev` and the Agent Chat UI |
| `requirements.txt` | Core project dependencies |

---

## 🤖 Agent Deep Dive (`agent.py`)

### Models

The system leverages OpenAI-compatible endpoints with dual-provider capability:

| Identifier | Provider | Default Model | Purpose |
|---|---|---|---|
| `FAST` | DigitalOcean AI | `openai-gpt-oss-20b` | Standard chat, quiz generation, and notebook organizing |
| `SMART` | Alibaba Cloud | `qwen-plus` | Complex or extended conversations (>12 messages) |
| `VISION` | Alibaba Cloud | `qwen3-vl-plus` | Multimodal screenshot and diagram understanding |

### Tools

| Tool | Description |
|---|---|
| `get_time` | Returns the current date and time |
| `get_profile` | Retrieves student name and proficiency level from runtime context |
| `save_note` | Persists a note to the user's private notebook store and updates chat state |
| `list_notes` | Lists all saved notes with their assigned topic notebooks |
| `create_flashcards` | Generates a structured deck of flashcard question/answer pairs |
| `make_quiz` | Invokes the quiz-maker sub-agent to generate a 3-question quiz with an answer key |

### Middleware Stack

Applied sequentially on every model invocation:

1. **`personalize`**: Dynamically tailors the system prompt to the user's name and experience level (beginner, intermediate, advanced).
2. **`ModelRouter`**: Automatically upgrades the active model from `FAST` to `SMART` once a conversation exceeds 12 messages.
3. **`SummarizationMiddleware`**: Triggers summarization when message tokens exceed 4,000, retaining the most recent 10 messages for continuous context.

### Context & State

- **`Context`**: Read-only runtime context passed on invoke (`user_name`, `level`, `user_id`).
- **`StudyState`**: State schema stored in checkpoints; tools return `Command(update=...)` to update state fields.

### Multi-Agent Architecture

- **Quiz Sub-Agent (`build_quiz_tool`)**: An isolated agent with its own system prompt wrapped as a tool. It formats 3 multiple-choice questions followed by an answers delimiter (`---ANSWERS---`).
- **Organizer Agent (`organizer.py`)**: A standalone agent invoked when notes need clustering. It analyzes note contents and existing notebooks, returning strict JSON mapping each note ID to a topic.

---

## 🃏 Flashcard System (`cards_store.py`)

1. **Deck Generation**: Request flashcards in chat (e.g., *"Make flashcards for Docker fundamentals"*). The agent calls `create_flashcards` to save the deck.
2. **Interactive 3D Flip Card**: Built with CSS 3D transforms (`rotateY`) inside the Streamlit view. Clicking flips between question and answer.
3. **Progress Tracking**: Users mark cards as ✅ **Got it** or ❌ **Missed it**. The app records right/wrong counts and last attempt status.
4. **Targeted Review**: Study all cards or filter down to **"Only the ones I missed"** to reinforce weak points.

---

## 🔊 Text-to-Speech System (`tts.py`)

- **Synthesizer**: Uses Alibaba Qwen-TTS (`qwen3-tts-flash` / `qwen3-tts-instruct-flash`).
- **Text Normalization**: Strips markdown, emojis, URLs, and code blocks to generate natural-sounding speech.
- **Smart Chunking**: Splits text into sentence-aligned pieces under 450 characters, then joins output audio using `wave`.
- **Audio Caching**: Audio files are cached to `audio_cache/` using an MD5 hash of voice, style, and text — subsequent replays are instantaneous and consume zero API quota.
- **Voices**: `Cherry`, `Serena`, `Ethan`, `Chelsie`, `Momo`, `Vivian`.
- **Styles**: Default voice, Friendly teacher, Calm and clear, Energetic coach.

---

## 🖼️ Multimodal Vision Processing

1. Upload any PNG or JPEG file (up to 5 MB) directly in the chat bar.
2. File validation verifies binary magic bytes (`\xff\xd8\xff` for JPEG, `\x89PNG` for PNG) rather than trusting file extensions alone.
3. Image bytes are processed entirely in memory via `describe_image_bytes()` using Alibaba Vision (`qwen3-vl-plus`).
4. The generated visual description is passed to the tutor alongside your prompt.

---

## 🔒 User Privacy & Session Management

- Every student session is identified by a 32-character hexadecimal key in the URL (`?u=<uid>`).
- Path validation enforces strict regex checks (`^[a-f0-9]{32}$`) to prevent directory traversal.
- User files are strictly isolated under `data/<uid>/`:
  - `notes.json`: User notes and notebook categorizations
  - `flashcards.json`: Decks and progress statistics
  - `chats.json`: Chat session history and metadata
- Checkpoint data is stored per thread in an `aiosqlite` database or in-memory fallback.

---

## 🔧 Setup & Installation

### 1. Prerequisites

- Python 3.11+
- DigitalOcean AI API Key (`DO_API_KEY`) — required for core models
- Alibaba Cloud API Key (`ALIBABA_API_KEY`) — optional, enables `SMART` model, Vision, and TTS

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create `.streamlit/secrets.toml` or a `.env` file in the project directory:

```toml
DO_API_KEY = "your-digitalocean-api-key"
ALIBABA_API_KEY = "your-alibaba-api-key"   # optional
```

You can optionally configure custom model endpoints:

```env
FAST_MODEL=openai-gpt-oss-20b
SMART_MODEL=qwen-plus
VISION_MODEL=qwen3-vl-plus
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501`. A unique user link (`?u=...`) will be generated automatically.

### 5. CLI Mode

To interact with the agent from a terminal without a browser:

```bash
python agent.py
```

---

## 💡 Example Prompts

```
"Explain how Retrieval-Augmented Generation (RAG) works in simple terms."
"Make me flashcards on Python list comprehensions and generators."
"Quiz me on relational database normalization."
"Save a note: Latency measures response time, while throughput measures capacity."
"What does this diagram represent?" (attach an architecture diagram)
```

---

## 📦 Dependencies

```
streamlit>=1.43
langchain>=1.0
langgraph>=1.0
langchain-openai
langgraph-checkpoint-sqlite
aiosqlite
httpx
python-dotenv
langgraph-cli[inmem]
```

---

## 📜 License

For educational and demonstration purposes.
