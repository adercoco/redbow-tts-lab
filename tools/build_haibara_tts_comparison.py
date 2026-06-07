#!/usr/bin/env python3
"""Build a focused Haibara-style comparison page."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HAIBARA_AUDIO = Path("/Users/ader/Documents/Codex/2026-05-23/youtube/extracted_audio/haibara")
OUT_DIR = ROOT / "haibara_tts_comparison"


def latest_haibara_run() -> Path:
    runs = sorted((ROOT / "tts_runs").glob("*/haibara_style-f5-cli"), reverse=True)
    if not runs:
        raise RuntimeError("No haibara_style-f5-cli runs found")
    return runs[0]


def audio(path: Path, base: Path) -> str:
    rel = os.path.relpath(path, base)
    return f'<audio controls preload="metadata" src="{html.escape(rel)}"></audio>'


def main() -> int:
    run_dir = latest_haibara_run()
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_files = [
        HAIBARA_AUDIO / "haibara_train_line_01.wav",
        HAIBARA_AUDIO / "haibara_train_line_02.wav",
        HAIBARA_AUDIO / "haibara_train_speaking_1520_2540.wav",
        HAIBARA_AUDIO / "haibara_forest_full.wav",
    ]
    ref_rows = []
    for path in reference_files:
        if path.exists():
            ref_rows.append(
                "<tr>"
                f"<td>{html.escape(path.name)}</td>"
                f"<td>{audio(path, OUT_DIR)}</td>"
                '<td contenteditable="true"></td>'
                "</tr>"
            )

    gen_rows = []
    for item in results:
        output = Path(item["output"]) if item.get("output") else None
        ref = Path(item["reference_audio"]) if item.get("reference_audio") else None
        gen_rows.append(
            "<tr>"
            f"<td>{html.escape(item['target_name'])}</td>"
            f"<td>{html.escape(item['goal'])}</td>"
            f"<td>{audio(ref, OUT_DIR) if ref and ref.exists() else ''}</td>"
            f"<td>{html.escape(item['text'])}</td>"
            f"<td>{audio(output, OUT_DIR) if output and output.exists() else ''}</td>"
            f"<td>{item['seconds']:.1f}s</td>"
            '<td contenteditable="true"></td>'
            "</tr>"
        )

    report = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>灰原氣質 TTS 比較</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", sans-serif; margin: 24px; background: #fbfaf8; color: #171312; }}
    h1, h2 {{ margin-bottom: 6px; }}
    p {{ color: #645c57; max-width: 980px; line-height: 1.55; }}
    table {{ width: 100%; border-collapse: collapse; background: white; margin: 16px 0 30px; }}
    th, td {{ border: 1px solid #ddd8d2; padding: 10px; vertical-align: top; }}
    th {{ background: #f1ede8; text-align: left; }}
    audio {{ width: 240px; }}
  </style>
</head>
<body>
  <h1>灰原氣質 TTS 比較</h1>
  <p>上半部是你資料夾裡的灰原片段，只做人工聽感參考。下半部是三種不複製真人聲紋的原創 F5-TTS 聲線方向，用來判斷哪個方向最接近「冷靜少女 / 疏離 / 清楚」的角色氣質。</p>

  <h2>灰原聽感參考</h2>
  <table>
    <thead><tr><th>檔案</th><th>音訊</th><th>人工觀察</th></tr></thead>
    <tbody>{''.join(ref_rows)}</tbody>
  </table>

  <h2>F5-TTS 原創聲線版本</h2>
  <table>
    <thead>
      <tr>
        <th>版本</th>
        <th>目標</th>
        <th>原創 Reference</th>
        <th>台詞</th>
        <th>TTS 輸出</th>
        <th>耗時</th>
        <th>人工評語</th>
      </tr>
    </thead>
    <tbody>{''.join(gen_rows)}</tbody>
  </table>
</body>
</html>
"""
    output = OUT_DIR / "index.html"
    output.write_text(report, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
