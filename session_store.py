"""Persistent JSON store for chat messages and agent notes.
Allows conversations and agent state notes to survive page refreshes and server restarts.
"""

import json
import threading
from pathlib import Path

PATH = Path(__file__).parent / "sessions.json"
_lock = threading.Lock()


def _load_all() -> dict:
    if not PATH.exists():
        return {}
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_all(data: dict) -> None:
    try:
        PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def get_session(thread_id: str) -> dict:
    with _lock:
        data = _load_all()
        return data.get(thread_id, {"messages": [], "notes": []})


def save_session(thread_id: str, messages: list, notes: list) -> None:
    with _lock:
        data = _load_all()
        data[thread_id] = {"messages": messages, "notes": notes}
        _save_all(data)


def clear_session(thread_id: str) -> None:
    with _lock:
        data = _load_all()
        if thread_id in data:
            data.pop(thread_id, None)
            _save_all(data)
