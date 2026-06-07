#!/usr/bin/env python3
"""Build no-reference Taiwan female voice-design audition report."""

from __future__ import annotations

import base64
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "teacher_model_audition_v4_no_reference"
INDEX = OUT / "index.html"
STANDALONE = OUT / "teacher_model_audition_v4_no_reference_standalone.html"
RESULT_FILES = ["qwen3_no_reference_results.json", "cosyvoice_instruct_short_no_reference_results.json"]
DESIGN_ORDER = [
    "taiwan_soft_low_r",
    "taipei_daily_pretty",
    "ntu_smart_gentle",
    "bookstore_senpai_no_ref",
    "soft_app_assistant",
    "calm_detective_girl",
]
MODEL_ORDER = ["Qwen3 VoiceDesign", "CosyVoice-300M-Instruct"]
TEXT_ORDER = ["tw_wait", "tw_weird", "tw_slow", "tw_tail"]


def esc(value: object) -> str:
    return html.escape(str(value))


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in RESULT_FILES:
        path = OUT / name
        if path.exists():
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def audio_src(output: str, standalone: bool) -> str:
    path = Path(output)
    if standalone:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:audio/wav;base64,{data}"
    return path.relative_to(OUT).as_posix()


def render_audio(row: dict[str, object], standalone: bool) -> str:
    output = str(row.get("output") or "")
    if not output or not Path(output).exists():
        return f'<p class="error">{esc(row.get("error", "no output"))}</p>'
    return f'<audio controls preload="metadata" src="{esc(audio_src(output, standalone))}"></audio>'


def render(standalone: bool) -> str:
    rows = load_rows()
    ok_rows = [r for r in rows if r.get("output") and Path(str(r["output"])).exists()]

    by_model = defaultdict(list)
    for row in ok_rows:
        by_model[str(row["family"])].append(float(row["seconds"]))

    cards = []
    for model in MODEL_ORDER:
        values = by_model.get(model, [])
        avg = statistics.mean(values) if values else 0
        cards.append(
            f"""
            <div class="summary-card">
              <b>{esc(model)}</b>
              <span>{len(values)} clips</span>
              <strong>{avg:.1f}s</strong>
              <small>無外部 reference audio</small>
            </div>
            """
        )
    for model, note in [
        ("F5-TTS", "本地 CLI 需要 ref_audio/ref_text；不支援純文字指定台灣女生聲音。"),
        ("GPT-SoVITS v2", "官方推論入口需要 ref_audio/ref_text；沒有 voice-design prompt。"),
        ("CosyVoice2-0.5B", "本地 CosyVoice2 zero-shot/instruct2 需要 prompt_wav；spk2info 沒有內建 speaker。"),
    ]:
        cards.append(
            f"""
            <div class="summary-card unsupported">
              <b>{esc(model)}</b>
              <span>not generated</span>
              <strong>不支援</strong>
              <small>{esc(note)}</small>
            </div>
            """
        )

    by_design_model = defaultdict(list)
    for row in rows:
        by_design_model[(str(row["design_id"]), str(row["family"]))].append(row)

    sections = []
    for design_id in DESIGN_ORDER:
        design_rows = [r for r in rows if r.get("design_id") == design_id]
        if not design_rows:
            continue
        first = design_rows[0]
        sections.append(
            f"""
            <section class="design">
              <div class="design-head">
                <div>
                  <p class="eyebrow">{esc(design_id)}</p>
                  <h2>{esc(first["label"])}</h2>
                </div>
                <p>{esc(first["prompt"])}</p>
              </div>
              <div class="model-grid">
            """
        )
        for model in MODEL_ORDER:
            model_rows = sorted(
                by_design_model[(design_id, model)],
                key=lambda r: TEXT_ORDER.index(str(r["text_id"])) if str(r["text_id"]) in TEXT_ORDER else 99,
            )
            sections.append(f'<article class="model"><h3>{esc(model)}</h3>')
            for row in model_rows:
                sections.append(
                    f"""
                    <section class="sample">
                      <div class="sample-head"><b>{esc(row["text_id"])}</b><span>{float(row["seconds"]):.1f}s</span></div>
                      <p>{esc(row["text"])}</p>
                      <small>{esc(row["mode"])}</small>
                      {render_audio(row, standalone)}
                    </section>
                    """
                )
            sections.append("</article>")
        sections.append("</div></section>")

    offline = (
        "<p class=\"lead\"><b>單檔離線版：</b>所有音檔已嵌入 HTML，下載到手機後不用同 Wi-Fi、不用 server。</p>"
        if standalone
        else "<p class=\"lead\">本機版使用相對音檔路徑；跨網路手機使用 standalone 單檔版。</p>"
    )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>無 Reference 台灣女生 TTS 試聽 v4</title>
  <style>
    :root {{
      --bg:#f7f8fb; --panel:#fff; --ink:#17181c; --muted:#5f6977;
      --line:#d9dee8; --accent:#b0182b; --soft:#f1f4f8;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; background:var(--bg); color:var(--ink);
      font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif;
      line-height:1.55;
    }}
    header, main {{ max-width:1240px; margin:0 auto; padding:16px 12px; }}
    h1 {{ margin:0 0 8px; font-size:30px; line-height:1.15; letter-spacing:0; }}
    h2 {{ margin:2px 0 4px; font-size:21px; letter-spacing:0; }}
    h3 {{ margin:0 0 8px; color:var(--accent); font-size:17px; letter-spacing:0; }}
    p {{ margin:0; }}
    audio {{ width:100%; height:38px; margin-top:8px; }}
    .lead {{ color:var(--muted); max-width:980px; margin-top:8px; }}
    .summary {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin-top:14px; }}
    .summary-card, .design, .model {{
      background:var(--panel); border:1px solid var(--line); border-radius:8px;
    }}
    .summary-card {{ padding:12px; display:grid; gap:4px; }}
    .summary-card b {{ color:var(--accent); }}
    .summary-card span, .summary-card small, .design-head p, .sample small {{ color:var(--muted); }}
    .summary-card strong {{ font-size:21px; }}
    .unsupported {{ opacity:.82; }}
    .design {{ margin-top:14px; overflow:hidden; }}
    .design-head {{
      display:grid; grid-template-columns:minmax(220px,320px) 1fr; gap:12px;
      padding:14px; border-bottom:1px solid var(--line); background:var(--soft);
    }}
    .eyebrow {{ color:var(--accent); font-weight:800; font-size:12px; }}
    .model-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; padding:10px; }}
    .model {{ padding:12px; }}
    .sample {{ padding-top:10px; margin-top:10px; border-top:1px solid var(--line); }}
    .sample:first-of-type {{ margin-top:0; }}
    .sample-head {{ display:flex; justify-content:space-between; gap:8px; color:var(--accent); font-size:13px; }}
    .sample p {{ margin:8px 0 2px; font-weight:650; }}
    .error {{ color:#9b1c31; font-weight:700; }}
    @media (max-width:960px) {{
      h1 {{ font-size:26px; }}
      .summary, .model-grid, .design-head {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>無 Reference 台灣女生 TTS 試聽 v4</h1>
    {offline}
    <p class="lead">這版不餵任何外部 reference audio。目標是測模型自己能不能靠文字條件做出「台灣、低卷舌、溫柔、清亮、漂亮耐聽」的年輕女生聲音。CosyVoice 已改用官方格式的短指令並加上 <code>&lt;|endofprompt|&gt;</code>，避免把角色 prompt 念進語音內容。F5-TTS、GPT-SoVITS v2、CosyVoice2 不能用這種方式指定聲音，所以只列能力限制，不混入 clone 結果。</p>
    <div class="summary">{''.join(cards)}</div>
  </header>
  <main>{''.join(sections)}</main>
</body>
</html>
"""


def main() -> int:
    INDEX.write_text(render(False), encoding="utf-8")
    STANDALONE.write_text(render(True), encoding="utf-8")
    rows = load_rows()
    ok = [r for r in rows if r.get("output") and Path(str(r["output"])).exists()]
    print(INDEX)
    print(STANDALONE)
    print(f"rows={len(rows)} audio={len(ok)} standalone_mb={STANDALONE.stat().st_size / 1024 / 1024:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
