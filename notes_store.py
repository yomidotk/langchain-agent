"""Study notes saved in notes.json, shared by every chat and kept across restarts.
Each note: {"id", "text", "topic"}. The topic (notebook) is filled in by the organizer agent."""

import json
import threading
import uuid
from pathlib import Path

PATH = Path(__file__).parent / "notes.json"
_lock = threading.Lock()


def _write(notes: list[dict]) -> None:
    PATH.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")


def load() -> list[dict]:
    try:
        raw = json.loads(PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    migrated = any(isinstance(n, str) for n in raw)  # older versions saved plain strings
    notes = [{"id": uuid.uuid4().hex[:8], "text": n, "topic": None} if isinstance(n, str) else n for n in raw]
    if migrated:
        with _lock:
            _write(notes)
    return notes


def texts() -> list[str]:
    return [n["text"] for n in load()]


def add(text: str) -> dict:
    with _lock:
        notes = load()
        note = {"id": uuid.uuid4().hex[:8], "text": text, "topic": None}  # topic stays empty until organized
        notes.append(note)
        _write(notes)
    return note


def delete(note_id: str) -> None:
    with _lock:
        _write([n for n in load() if n["id"] != note_id])


def set_topics(mapping: dict[str, str]) -> None:
    """mapping: note id -> notebook title"""
    with _lock:
        notes = load()
        for n in notes:
            if n["id"] in mapping:
                n["topic"] = mapping[n["id"]]
        _write(notes)
