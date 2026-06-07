#!/usr/bin/env python3
"""Create owned synthetic reference clips for model plumbing tests.

These are not character clones. They are locally generated references so we can
verify F5-TTS end to end before adding authorized human reference audio.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "tools" / "tts_eval_config.json"
REF_DIR = ROOT / "reference_audio" / "synthetic"

VOICE_BY_TARGET = {
    "bow-detective": ("Tingting", 205),
    "taiwan-variety": ("Meijia", 198),
    "news-anchor": ("Meijia", 172),
    "warm-narrator": ("Meijia", 160),
    "dramatic-reveal": ("Tingting", 142),
}


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    REF_DIR.mkdir(parents=True, exist_ok=True)

    for target in config["targets"]:
        text = target["tests"][0]
        voice, rate = VOICE_BY_TARGET.get(target["id"], ("Meijia", 180))
        wav_path = REF_DIR / f"{target['id']}.wav"
        with tempfile.TemporaryDirectory(prefix="red-bow-ref-") as tmp:
            aiff_path = Path(tmp) / "ref.aiff"
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff_path), text], check=True)
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1", str(aiff_path), str(wav_path)],
                check=True,
            )
        target["reference_audio"] = str(wav_path.relative_to(ROOT))
        target["reference_text"] = text

    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote references to {REF_DIR}")
    print(f"Updated {CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
