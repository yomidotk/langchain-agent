"""Tiny JSON store for flashcard decks + progress, so cards survive restarts (so you can redo them later)."""

import json
import threading
import uuid
from pathlib import Path

PATH = Path(__file__).parent / "flashcards.json"
_lock = threading.Lock()


def load() -> dict:
    """{concept: [{"id", "question", "answer", "right", "wrong", "last"}]}  (last = "right" | "wrong" | None)"""
    if not PATH.exists():
        return {}
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(data: dict) -> None:
    PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def add_cards(concept: str, cards: list[dict]) -> int:
    concept = concept.strip().title() or "General"
    with _lock:
        data = load()
        deck = data.setdefault(concept, [])
        for c in cards:
            deck.append(
                {"id": uuid.uuid4().hex[:8], "question": c["question"], "answer": c["answer"],
                 "right": 0, "wrong": 0, "last": None}
            )
        _save(data)
    return len(cards)


def record(concept: str, card_id: str, correct: bool) -> None:
    with _lock:
        data = load()
        for c in data.get(concept, []):
            if c["id"] == card_id:
                c["right" if correct else "wrong"] += 1
                c["last"] = "right" if correct else "wrong"
        _save(data)


def delete_deck(concept: str) -> None:
    with _lock:
        data = load()
        data.pop(concept, None)
        _save(data)
