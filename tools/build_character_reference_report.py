#!/usr/bin/env python3
"""Build a local listening report from downloaded character clips.

This report is for manual listening and style analysis only. It does not feed
copyrighted or actor-performed audio into a cloning model.
"""

from __future__ import annotations

import html
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = Path("/Users/ader/Documents/Codex/2026-05-23/youtube/clips")
OUT_DIR = ROOT / "character_reference_report"

ROLE_TRAITS = {
    "灰原哀": ("冷靜少女", "低情緒、清楚、偏冷、短句有距離"),
    "阿笠博士": ("發明博士", "年長、溫和、帶喜感、語速中慢"),
    "毛利小五郎": ("糊塗偵探", "中年男性、誇張、喜劇感、音量起伏大"),
    "服部平次 / 遠山和葉": ("熱血少年 / 關西少女", "速度快、情緒外放、互動感強"),
    "鈴木園子 / 毛利蘭 topic": ("清亮女主 / 活潑同學", "年輕女性、亮、自然、口語互動"),
    "目暮十三 voice sample": ("沉穩警部", "中低音、穩、權威、語速偏慢"),
    "高木涉": ("年輕刑警", "青年男性、緊張感、自然台詞"),
    "怪盜基德 / 柯南": ("優雅怪盜 / 紅領結偵探", "一高一低對比，怪盜更從容，偵探更亮更快"),
}


def parse_index() -> list[dict]:
    index = CLIPS_DIR / "INDEX.md"
    rows: list[dict] = []
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or "`" not in line or "youtube.com" not in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 3:
            continue
        role = parts[0]
        file_match = re.search(r"`([^`]+)`", parts[1])
        source_match = re.search(r"https://www\.youtube\.com/watch\?v=[^)\\s]+", parts[2])
        if not file_match or not source_match:
            continue
        path = CLIPS_DIR / file_match.group(1)
        if not path.exists():
            continue
        style, traits = ROLE_TRAITS.get(role, ("待標註", "請人工聽完後補上"))
        rows.append(
            {
                "role": role,
                "style": style,
                "traits": traits,
                "path": path,
                "source": source_match.group(0),
                "duration": duration(path),
            }
        )
    return rows


def duration(path: Path) -> str:
    try:
        output = subprocess.check_output(
            ["mdls", "-name", "kMDItemDurationSeconds", "-raw", str(path)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        seconds = float(output)
        return f"{seconds:.1f}s"
    except Exception:
        return ""


def build_report(rows: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for row in rows:
        rel = os.path.relpath(row["path"], OUT_DIR)
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['role'])}</td>"
            f"<td>{html.escape(row['style'])}</td>"
            f"<td>{html.escape(row['traits'])}</td>"
            f"<td>{html.escape(row['duration'])}</td>"
            f'<td><video controls preload="metadata" src="{html.escape(rel)}"></video></td>'
            f'<td><a href="{html.escape(row["source"])}">YouTube</a></td>'
            '<td contenteditable="true"></td>'
            "</tr>"
        )

    report = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>角色聲線參考表</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", sans-serif; margin: 24px; background: #fbfaf8; color: #191614; }}
    h1 {{ margin-bottom: 4px; }}
    .note {{ max-width: 980px; color: #6f6762; line-height: 1.5; }}
    table {{ width: 100%; border-collapse: collapse; background: white; margin-top: 18px; }}
    th, td {{ border: 1px solid #ddd8d2; padding: 10px; vertical-align: top; }}
    th {{ background: #f1ede8; text-align: left; }}
    video {{ width: 280px; max-width: 36vw; }}
  </style>
</head>
<body>
  <h1>角色聲線參考表</h1>
  <p class="note">這份表只做人工聽感分析。不要把未授權角色、配音員、藝人或網紅聲音拿去做 voice cloning。可用方向是把聽到的非聲紋特徵轉成原創聲線：音高、語速、情緒、咬字、停頓。</p>
  <table>
    <thead>
      <tr>
        <th>角色/主題</th>
        <th>原創聲線方向</th>
        <th>初步聲線特徵</th>
        <th>長度</th>
        <th>片段</th>
        <th>來源</th>
        <th>人工標註</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output = OUT_DIR / "index.html"
    output.write_text(report, encoding="utf-8")
    return output


def main() -> int:
    rows = parse_index()
    output = build_report(rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
