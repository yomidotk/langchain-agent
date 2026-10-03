"""One-time helper: move the data from the old shared version into YOUR private space.

1. Open the app and copy the key from the address bar: ...?u=THIS_PART
2. Run:  python claim_old_data.py THIS_PART
"""

import shutil
import sys
from pathlib import Path

import userdata

if len(sys.argv) != 2 or not userdata.is_valid(sys.argv[1]) or sys.argv[1] == "local":
    sys.exit("Usage: python claim_old_data.py <the 32-character key from ?u=... in the app URL>")

here, dest = Path(__file__).parent, userdata.user_dir(sys.argv[1])
for name in ("notes.json", "flashcards.json", "chats.json"):
    src = here / name
    if not src.exists():
        continue
    if (dest / name).exists():
        print(f"skipped {name}: you already have one in your private space (rename it first to replace it)")
        continue
    shutil.move(str(src), dest / name)
    print(f"moved {name} -> {dest / name}")
print("Done. Refresh the app.")
