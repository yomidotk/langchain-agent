"""A second agent whose only job is to sort the student's notes into notebooks by topic."""

import json
import re

from langchain.agents import create_agent

import agent as core  # for the model (core.FAST)
import notes_store

SYSTEM = (
    "You organize a student's study notes into notebooks by topic.\n"
    "You get the existing notebook titles and a list of notes as 'id | text'.\n"
    "Rules:\n"
    "- Every note goes in exactly one notebook.\n"
    "- Reuse an existing notebook title whenever a note fits it. Only create a new notebook when none fits.\n"
    "- Notebooks are broad topics (for example 'AI Fine-tuning', 'LangChain', 'Tech Tools'), never one per note.\n"
    "- Aim for 2 to 8 notebooks. Put strays in a notebook called 'General'.\n"
    "- Titles are at most 4 words.\n"
    'Reply with ONLY JSON, no other text: {"notebooks": [{"title": "...", "note_ids": ["id1", "id2"]}]}'
)


def _parse(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("the organizer didn't return any notebooks")
    return json.loads(match.group(0))["notebooks"]


async def organize_notes(fresh: bool = False) -> int:
    """Sort every note into a notebook. fresh=True ignores the current notebooks and regroups from scratch.
    Returns the number of notebooks."""
    notes = notes_store.load()
    if not notes:
        return 0
    existing = [] if fresh else sorted({n["topic"] for n in notes if n.get("topic")})
    listing = "\n".join(f"{n['id']} | {n['text']}" for n in notes)
    request = f"Existing notebooks: {existing or 'none yet'}\n\nNotes (id | text):\n{listing}"

    organizer = create_agent(model=core.FAST, tools=[], system_prompt=SYSTEM)
    result = await organizer.ainvoke({"messages": [{"role": "user", "content": request}]})
    notebooks = _parse(result["messages"][-1].text)

    ids = {n["id"] for n in notes}
    mapping: dict[str, str] = {}
    for nb in notebooks:
        title = str(nb.get("title", "")).strip()[:40] or "General"
        for note_id in nb.get("note_ids", []):
            if note_id in ids and note_id not in mapping:
                mapping[note_id] = title
    for note_id in ids - mapping.keys():  # anything the model forgot
        mapping[note_id] = "General"
    notes_store.set_topics(mapping)
    return len(set(mapping.values()))
