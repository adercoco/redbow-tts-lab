#!/usr/bin/env python3
"""Create original synthetic references for Haibara-style auditions."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reference_audio" / "original_haibara_style"

REFERENCES = [
    (
        "cool_dry.wav",
        "Meijia",
        156,
        "我記得這應該是感冒藥。吃下去之後，麻煩你在一旁安分點。",
    ),
    (
        "soft_low.wav",
        "Meijia",
        136,
        "你快走。這裡已經不安全了，不要再回頭。",
    ),
    (
        "clear_young.wav",
        "Tingting",
        168,
        "等一下，這裡有一個很奇怪的地方。",
    ),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, voice, rate, text in REFERENCES:
        output = OUT_DIR / filename
        with tempfile.TemporaryDirectory(prefix="red-bow-haibara-style-") as tmp:
            aiff = Path(tmp) / "ref.aiff"
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1", str(aiff), str(output)],
                check=True,
            )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
