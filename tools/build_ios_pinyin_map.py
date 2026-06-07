#!/usr/bin/env python3
"""Build a compact offline Chinese-character to pinyin map for the iOS app."""

from __future__ import annotations

import json
from pathlib import Path

from pypinyin import Style, pinyin


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "RedBowVoice" / "RedBowAssets" / "pinyin_map.json"


def pinyin_for_char(ch: str) -> str | None:
    value = pinyin(
        ch,
        style=Style.TONE3,
        heteronym=False,
        neutral_tone_with_five=True,
        errors=lambda chars: list(chars),
    )[0][0]
    value = value.replace("ü", "v").lower()
    if value == ch:
        return None
    if all(c.isascii() and (c.isalpha() or c.isdigit()) for c in value):
        return value
    return None


def main() -> int:
    mapping: dict[str, str] = {}
    for codepoint in range(0x4E00, 0xA000):
        ch = chr(codepoint)
        value = pinyin_for_char(ch)
        if value:
            mapping[ch] = value

    OUT.write_text(
        json.dumps(mapping, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"{OUT} {len(mapping)} entries {OUT.stat().st_size / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
