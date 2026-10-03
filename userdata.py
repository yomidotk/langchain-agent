"""Every visitor gets a private folder: data/<key>/. The key is a random 32-character id that lives in the URL."""

import re
import uuid
from pathlib import Path

DATA = Path(__file__).parent / "data"
_VALID = re.compile(r"^[a-f0-9]{32}$|^local$")  # strict on purpose: stops tricks like ?u=../../something


def new_uid() -> str:
    return uuid.uuid4().hex


def is_valid(uid) -> bool:
    return isinstance(uid, str) and bool(_VALID.match(uid))


def user_dir(uid: str) -> Path:
    if not is_valid(uid):
        raise ValueError("invalid user id")
    folder = DATA / uid
    folder.mkdir(parents=True, exist_ok=True)
    return folder
