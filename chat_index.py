"""Remembers which chats exist and which one you used last (saved in chats.json).
The messages themselves live in studeno.db (the agent's checkpointer)."""

import json
import time
import uuid
from pathlib import Path

PATH = Path(__file__).parent / "chats.json"


def load() -> dict:
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data.setdefault("last", None)
    data.setdefault("chats", {})
    return data


def _save(data: dict) -> None:
    PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def new_id() -> str:
    return uuid.uuid4().hex


def set_last(chat_id: str) -> None:
    data = load()
    if data["last"] != chat_id:
        data["last"] = chat_id
        _save(data)


def touch(chat_id: str, title: str | None = None) -> None:
    """Register a chat (on its first message) and bump its 'updated' time."""
    data = load()
    chat = data["chats"].setdefault(chat_id, {"title": (title or "New chat")[:40], "updated": 0})
    chat["updated"] = time.time()
    data["last"] = chat_id
    _save(data)


def recent(limit: int = 15) -> list[tuple[str, str]]:
    chats = load()["chats"]
    ordered = sorted(chats.items(), key=lambda kv: kv[1]["updated"], reverse=True)
    return [(cid, c["title"]) for cid, c in ordered[:limit]]
