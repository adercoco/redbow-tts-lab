#!/usr/bin/env python3
"""Rebuild the teacher-model audition HTML from generated result JSON files."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "teacher_model_audition_v2"


def esc(value: object) -> str:
    return html.escape(str(value))


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for filename in ["results.json", "cosyvoice_results.json", "gpt_sovits_results.json"]:
        path = OUT / filename
        if path.exists():
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def audio_rel(path_text: str) -> str:
    return Path(path_text).relative_to(OUT).as_posix()


def render_report(rows: list[dict[str, object]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["family"]), str(row["candidate_id"])), []).append(row)

    sections = []
    for (_family, _candidate_id), items in grouped.items():
        first = items[0]
        sections.append(
            f"""
            <article class="candidate">
              <div class="family">{esc(first['family'])}</div>
              <h2>{esc(first['label'])}</h2>
              <p class="prompt">{esc(first['prompt'])}</p>
            """
        )
        for row in items:
            sections.append(
                f"""
                <section class="sample">
                  <div class="sample-head"><b>{esc(row['text_id'])}</b><span>{float(row['seconds']):.1f}s</span></div>
                  <p>{esc(row['text'])}</p>
                """
            )
            if row.get("output"):
                sections.append(f'<audio controls preload="metadata" src="{esc(audio_rel(str(row["output"])))}"></audio>')
            else:
                sections.append(f'<p class="error">{esc(row.get("error", ""))}</p>')
            sections.append("</section>")
        sections.append("</article>")

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>台灣女生老師聲音候選</title>
  <style>
    :root {{
      --bg:#f7f8fb;
      --panel:#fff;
      --ink:#17181c;
      --muted:#5f6977;
      --line:#d9dee8;
      --accent:#b0182b;
      --soft:#f2f5f8;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      background:var(--bg);
      color:var(--ink);
      font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif;
      line-height:1.58;
    }}
    header, main {{ max-width:1120px; margin:0 auto; padding:16px 12px; }}
    h1 {{ margin:0 0 8px; font-size:32px; line-height:1.15; letter-spacing:0; }}
    h2 {{ margin:4px 0 8px; font-size:20px; letter-spacing:0; }}
    p {{ margin:0; }}
    .lead {{ color:var(--muted); font-size:16px; max-width:900px; }}
    .notice, .candidate {{
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:8px;
      padding:14px;
    }}
    .notice {{ margin-top:12px; }}
    .grid {{
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:10px;
      margin-top:12px;
    }}
    .family {{ color:var(--accent); font-weight:800; font-size:13px; }}
    .prompt {{ color:var(--muted); font-size:14px; }}
    .sample {{
      margin-top:10px;
      padding-top:10px;
      border-top:1px solid var(--line);
    }}
    .sample-head {{ display:flex; justify-content:space-between; gap:8px; color:var(--accent); font-size:13px; }}
    .sample p {{ margin:8px 0; font-weight:650; }}
    audio {{ width:100%; height:38px; }}
    .error {{ color:#9b1c31; }}
    @media (max-width:760px) {{
      h1 {{ font-size:28px; }}
      .grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>台灣女生老師聲音候選</h1>
    <p class="lead">目標：道地台灣國語、低卷舌、清亮溫柔、年輕女生。這頁包含 Qwen3 VoiceDesign 多角色聲線、F5-TTS、CosyVoice2、GPT-SoVITS v2，全部使用同一組測試句，方便直接聽聲音、口音和穩定度。</p>
    <section class="notice">
      <p><b>判斷標準：</b>先聽是否像台灣人中文，其次聽女聲是否清亮溫柔，再看是否有奇怪捲舌、兒化音、主播腔或過度娃娃音。勝出的聲音才拿去產老師語料，避免把不好的口音蒸餾進 ZipVoice。</p>
    </section>
  </header>
  <main class="grid">{''.join(sections)}</main>
</body>
</html>
"""


def main() -> int:
    rows = load_rows()
    (OUT / "index.html").write_text(render_report(rows), encoding="utf-8")
    print(OUT / "index.html")
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
