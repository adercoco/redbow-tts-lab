#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "LARGE_FILES_MANIFEST.json").read_text())
for item in manifest:
    target = ROOT / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        for part in item["parts"]:
            out.write((ROOT / part).read_bytes())
    print(f"restored {target.relative_to(ROOT)}")
