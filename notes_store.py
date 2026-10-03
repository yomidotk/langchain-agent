"""Study notes saved in notes.json, shared by every chat and kept across restarts."""

import json
from pathlib import Path

PATH = Path(__file__).parent / "notes.json"


def load() -> list[str]:
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def add(note: str) -> int:
    notes = load()
    notes.append(note)
    PATH.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(notes)
