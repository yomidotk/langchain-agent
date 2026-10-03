"""Remembers which chats a visitor has and which one they used last (data/<uid>/chats.json).
The messages themselves live in study_buddy.db (the agent's checkpointer)."""

import json
import time
import uuid

from userdata import user_dir


def _path(uid: str):
    return user_dir(uid) / "chats.json"


def load(uid: str) -> dict:
    try:
        data = json.loads(_path(uid).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data.setdefault("last", None)
    data.setdefault("chats", {})
    return data


def _save(uid: str, data: dict) -> None:
    _path(uid).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def new_id() -> str:
    return uuid.uuid4().hex


def set_last(uid: str, chat_id: str) -> None:
    data = load(uid)
    if data["last"] != chat_id:
        data["last"] = chat_id
        _save(uid, data)


def touch(uid: str, chat_id: str, title: str | None = None) -> None:
    """Register a chat (on its first message) and bump its 'updated' time."""
    data = load(uid)
    chat = data["chats"].setdefault(chat_id, {"title": (title or "New chat")[:40], "updated": 0})
    chat["updated"] = time.time()
    data["last"] = chat_id
    _save(uid, data)


def recent(uid: str, limit: int = 15) -> list[tuple[str, str]]:
    chats = load(uid)["chats"]
    ordered = sorted(chats.items(), key=lambda kv: kv[1]["updated"], reverse=True)
    return [(cid, c["title"]) for cid, c in ordered[:limit]]
