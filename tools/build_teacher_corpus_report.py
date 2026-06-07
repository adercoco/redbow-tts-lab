#!/usr/bin/env python3
"""Build a review page for generated teacher corpus audio."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "distillation" / "taiwan_mandarin_low_r" / "teacher_qwen3_1p7b_seed"


def main() -> int:
    manifest_path = DEFAULT_CORPUS / "manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_rows = []
    for row in rows:
        audio = ""
        if row.get("audio"):
            rel = Path(row["audio"]).relative_to(DEFAULT_CORPUS).as_posix()
            audio = f'<audio controls preload="metadata" src="{html.escape(rel)}"></audio>'
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['id'])}</td>"
            f"<td>{html.escape(row.get('category', ''))}</td>"
            f"<td>{html.escape(row['text'])}</td>"
            f"<td>{audio}</td>"
            f"<td>{float(row['seconds']):.1f}s</td>"
            f"<td>{html.escape(row['status'])}</td>"
            f"<td>{html.escape(row.get('error', ''))}</td>"
            "</tr>"
        )

    report = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>taiwan_mandarin_low_r Teacher Corpus Seed</title>
<style>
body{{margin:0;background:#f7f4ef;color:#181514;font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif;}}
header{{background:#b51d2a;color:white;padding:18px 16px 12px;}}
h1{{font-size:22px;line-height:1.2;margin:0 0 6px;}}
header p{{margin:0;font-size:14px;opacity:.9;line-height:1.5;}}
.wrap{{overflow-x:auto;margin:14px;border:1px solid #ded6cc;background:white;}}
table{{border-collapse:collapse;min-width:880px;width:100%;}}
th,td{{border-bottom:1px solid #eee5dc;padding:10px;text-align:left;vertical-align:top;font-size:14px;}}
th{{background:#fff5ed;}}
audio{{width:230px;max-width:70vw;}}
.note{{padding:10px 16px;color:#5f5550;font-size:14px;line-height:1.6;}}
</style>
</head>
<body>
<header>
<h1>taiwan_mandarin_low_r Teacher Corpus Seed</h1>
<p>正式 teacher voice 的第一批蒸餾資料。請確認每句都維持台灣國語低卷舌、清亮溫柔、克制可愛。</p>
</header>
<p class="note">這不是 student 結果，是 1.7B teacher 原版資料。通過後再擴到數百/數千句，用來訓練小模型。</p>
<div class="wrap">
<table>
<thead><tr><th>ID</th><th>Category</th><th>Text</th><th>Teacher Audio</th><th>Gen Time</th><th>Status</th><th>Error</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
</div>
</body>
</html>
"""
    (DEFAULT_CORPUS / "index.html").write_text(report, encoding="utf-8")
    print(DEFAULT_CORPUS / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
