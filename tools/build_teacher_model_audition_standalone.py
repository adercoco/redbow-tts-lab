#!/usr/bin/env python3
"""Build a single-file, phone-portable HTML audition report with embedded audio."""

from __future__ import annotations

import base64
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "teacher_model_audition_v2"
INDEX = REPORT_DIR / "index.html"
OUT = REPORT_DIR / "teacher_model_audition_v2_standalone.html"


SRC_RE = re.compile(r'src="([^"]+\.wav)"')


def embed_audio(match: re.Match[str]) -> str:
    rel = match.group(1)
    wav_path = REPORT_DIR / rel
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)
    data = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    return f'src="data:audio/wav;base64,{data}"'


def main() -> int:
    html = INDEX.read_text(encoding="utf-8")
    html = SRC_RE.sub(embed_audio, html)
    html = html.replace(
        "<h1>台灣女生老師聲音候選</h1>",
        "<h1>台灣女生老師聲音候選</h1>\n"
        "    <p class=\"lead\"><b>單檔離線版：</b>音檔已嵌入 HTML，下載到手機後不需要同 Wi-Fi，也不需要本機 server。</p>",
    )
    OUT.write_text(html, encoding="utf-8")
    print(OUT)
    print(f"{OUT.stat().st_size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
