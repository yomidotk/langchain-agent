"""Flashcard decks + progress, saved per visitor in data/<uid>/flashcards.json (so you can redo them later)."""

import json
import threading
import uuid

from userdata import user_dir

_lock = threading.Lock()


def _path(uid: str):
    return user_dir(uid) / "flashcards.json"


def load(uid: str) -> dict:
    """{concept: [{"id", "question", "answer", "right", "wrong", "last"}]}  (last = "right" | "wrong" | None)"""
    try:
        return json.loads(_path(uid).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(uid: str, data: dict) -> None:
    _path(uid).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def add_cards(uid: str, concept: str, cards: list[dict]) -> int:
    concept = concept.strip().title() or "General"
    with _lock:
        data = load(uid)
        deck = data.setdefault(concept, [])
        for c in cards:
            deck.append(
                {"id": uuid.uuid4().hex[:8], "question": c["question"], "answer": c["answer"],
                 "right": 0, "wrong": 0, "last": None}
            )
        _save(uid, data)
    return len(cards)


def record(uid: str, concept: str, card_id: str, correct: bool) -> None:
    with _lock:
        data = load(uid)
        for c in data.get(concept, []):
            if c["id"] == card_id:
                c["right" if correct else "wrong"] += 1
                c["last"] = "right" if correct else "wrong"
        _save(uid, data)


def delete_deck(uid: str, concept: str) -> None:
    with _lock:
        data = load(uid)
        data.pop(concept, None)
        _save(uid, data)
