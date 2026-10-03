"""Study notes, saved per visitor in data/<uid>/notes.json and shared by that visitor's chats.
Each note: {"id", "text", "topic"}. The topic (notebook) is filled in by the organizer agent."""

import json
import threading
import uuid

from userdata import user_dir

_lock = threading.Lock()


def _path(uid: str):
    return user_dir(uid) / "notes.json"


def _write(uid: str, notes: list[dict]) -> None:
    _path(uid).write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")


def load(uid: str) -> list[dict]:
    try:
        raw = json.loads(_path(uid).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    migrated = any(isinstance(n, str) for n in raw)  # older versions saved plain strings
    notes = [{"id": uuid.uuid4().hex[:8], "text": n, "topic": None} if isinstance(n, str) else n for n in raw]
    if migrated:
        with _lock:
            _write(uid, notes)
    return notes


def add(uid: str, text: str) -> dict:
    with _lock:
        notes = load(uid)
        note = {"id": uuid.uuid4().hex[:8], "text": text, "topic": None}  # topic stays empty until organized
        notes.append(note)
        _write(uid, notes)
    return note


def delete(uid: str, note_id: str) -> None:
    with _lock:
        _write(uid, [n for n in load(uid) if n["id"] != note_id])


def set_topics(uid: str, mapping: dict[str, str]) -> None:
    """mapping: note id -> notebook title"""
    with _lock:
        notes = load(uid)
        for n in notes:
            if n["id"] in mapping:
                n["topic"] = mapping[n["id"]]
        _write(uid, notes)
