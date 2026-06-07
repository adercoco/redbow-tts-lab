#!/usr/bin/env python3
"""Build the v3 clone-model role-grid HTML report."""

from __future__ import annotations

import base64
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "teacher_model_audition_v3"
RESULTS = OUT / "clone_model_role_grid_results.json"
INDEX = OUT / "index.html"
STANDALONE = OUT / "teacher_model_audition_v3_standalone.html"

ROLE_ORDER = [
    "low_r_clean",
    "ntu_literature_warm",
    "coffee_shop_friend",
    "bookstore_senpai",
    "campus_radio_sister",
    "clinic_nurse_soft",
]
MODEL_ORDER = ["GPT-SoVITS v2", "CosyVoice2", "F5-TTS"]
TEXT_ORDER = ["tw_wait", "tw_detail", "tw_slow", "tw_weird"]


def esc(value: object) -> str:
    return html.escape(str(value))


def audio_rel(path_text: str) -> str:
    return Path(path_text).relative_to(OUT).as_posix()


def render_audio(row: dict[str, object], standalone: bool) -> str:
    output = str(row.get("output") or "")
    if not output:
        return f'<p class="error">{esc(row.get("error", ""))}</p>'
    if standalone:
        data = base64.b64encode(Path(output).read_bytes()).decode("ascii")
        src = f"data:audio/wav;base64,{data}"
    else:
        src = audio_rel(output)
    return f'<audio controls preload="metadata" src="{esc(src)}"></audio>'


def render(standalone: bool = False) -> str:
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    ok_rows = [r for r in rows if r.get("output") and Path(str(r["output"])).exists()]
    by_role_model: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_role_model[(str(row["role_id"]), str(row["family"]))].append(row)

    by_family = defaultdict(list)
    for row in ok_rows:
        by_family[str(row["family"])].append(float(row["seconds"]))

    summary_cards = []
    for family in MODEL_ORDER:
        values = by_family.get(family, [])
        avg = statistics.mean(values) if values else 0
        fastest = min(values) if values else 0
        slowest = max(values) if values else 0
        summary_cards.append(
            f"""
            <div class="summary-card">
              <b>{esc(family)}</b>
              <span>{len(values)} clips</span>
              <strong>{avg:.1f}s</strong>
              <small>avg / fastest {fastest:.1f}s / slowest {slowest:.1f}s</small>
            </div>
            """
        )

    provenance = f"""
      <section class="provenance">
        <h2>生成來源驗證</h2>
        <p>三個模型都使用同一批 Qwen3 台灣女生 reference，所以聲音相似是預期結果；如果 clone 做得好，本來就會往同一個角色聲線靠近。它們不是同一批音檔：輸出資料夾、檔案大小、hash、生成時間都不同。</p>
        <div class="prov-grid">
          <div><b>GPT-SoVITS v2</b><span>官方 GPT-SoVITS v2 底模，常駐載入一次後逐句 zero-shot；輸出在 <code>audio/gpt_sovits_v2/</code>。</span></div>
          <div><b>CosyVoice2</b><span>官方 CosyVoice2-0.5B zero-shot；輸出在 <code>audio/cosyvoice2/</code>。</span></div>
          <div><b>F5-TTS</b><span>F5TTS_v1_Base CLI zero-shot；輸出在 <code>audio/f5_tts/</code>。</span></div>
        </div>
      </section>
    """

    role_sections = []
    for role_id in ROLE_ORDER:
        role_rows = [r for r in rows if r.get("role_id") == role_id]
        if not role_rows:
            continue
        first = role_rows[0]
        role_sections.append(
            f"""
            <section class="role">
              <div class="role-head">
                <div>
                  <p class="eyebrow">{esc(role_id)}</p>
                  <h2>{esc(first["label"])}</h2>
                </div>
                <p>{esc(first["prompt"])}</p>
              </div>
              <div class="model-grid">
            """
        )
        for family in MODEL_ORDER:
            model_rows = sorted(
                by_role_model.get((role_id, family), []),
                key=lambda r: TEXT_ORDER.index(str(r["text_id"])) if str(r["text_id"]) in TEXT_ORDER else 99,
            )
            role_sections.append(
                f"""
                <article class="model">
                  <h3>{esc(family)}</h3>
                """
            )
            for row in model_rows:
                role_sections.append(
                    f"""
                    <section class="sample">
                      <div class="sample-head">
                        <b>{esc(row["text_id"])}</b>
                        <span>{float(row["seconds"]):.1f}s</span>
                      </div>
                      <p>{esc(row["text"])}</p>
                      {render_audio(row, standalone)}
                    </section>
                    """
                )
            role_sections.append("</article>")
        role_sections.append("</div></section>")

    offline_note = (
        '<p class="lead"><b>單檔離線版：</b>所有音檔已嵌入 HTML，下載到手機後不用同 Wi-Fi、不用 server。</p>'
        if standalone
        else '<p class="lead">本機版使用相對音檔路徑；若要跨網路用手機看，請開 standalone 單檔版。</p>'
    )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>台灣女生 Clone TTS 多角色試聽 v3</title>
  <style>
    :root {{
      --bg:#f6f7f9;
      --panel:#fff;
      --ink:#17181c;
      --muted:#606a78;
      --line:#d9dee8;
      --accent:#b0182b;
      --soft:#f0f3f7;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      background:var(--bg);
      color:var(--ink);
      font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif;
      line-height:1.55;
    }}
    header, main {{ max-width:1280px; margin:0 auto; padding:16px 12px; }}
    h1 {{ margin:0 0 8px; font-size:30px; line-height:1.15; letter-spacing:0; }}
    h2 {{ margin:2px 0 4px; font-size:22px; letter-spacing:0; }}
    h3 {{ margin:0 0 8px; font-size:17px; letter-spacing:0; color:var(--accent); }}
    p {{ margin:0; }}
    audio {{ width:100%; height:38px; }}
    .lead {{ color:var(--muted); max-width:980px; margin-top:8px; }}
    .summary {{
      display:grid;
      grid-template-columns:repeat(3,minmax(0,1fr));
      gap:10px;
      margin-top:14px;
    }}
    .summary-card, .role, .model {{
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:8px;
    }}
    .summary-card {{ padding:12px; display:grid; gap:3px; }}
    .summary-card b {{ color:var(--accent); }}
    .summary-card span, .summary-card small, .role-head p {{ color:var(--muted); }}
    .summary-card strong {{ font-size:24px; }}
    .provenance {{
      margin-top:14px;
      padding:14px;
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:8px;
    }}
    .provenance h2 {{ margin:0 0 6px; font-size:20px; }}
    .provenance p {{ color:var(--muted); }}
    .prov-grid {{
      display:grid;
      grid-template-columns:repeat(3,minmax(0,1fr));
      gap:10px;
      margin-top:12px;
    }}
    .prov-grid div {{
      display:grid;
      gap:5px;
      padding:10px;
      border:1px solid var(--line);
      border-radius:8px;
      background:var(--soft);
    }}
    .prov-grid b {{ color:var(--accent); }}
    .prov-grid span {{ color:var(--muted); font-size:14px; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.92em; }}
    .role {{ margin-top:14px; overflow:hidden; }}
    .role-head {{
      display:grid;
      grid-template-columns:minmax(220px,320px) 1fr;
      gap:12px;
      padding:14px;
      border-bottom:1px solid var(--line);
      background:var(--soft);
    }}
    .eyebrow {{ color:var(--accent); font-weight:800; font-size:12px; }}
    .model-grid {{
      display:grid;
      grid-template-columns:repeat(3,minmax(0,1fr));
      gap:10px;
      padding:10px;
    }}
    .model {{ padding:12px; }}
    .sample {{
      padding-top:10px;
      margin-top:10px;
      border-top:1px solid var(--line);
    }}
    .sample:first-of-type {{ margin-top:0; }}
    .sample-head {{
      display:flex;
      justify-content:space-between;
      gap:8px;
      color:var(--accent);
      font-size:13px;
    }}
    .sample p {{ margin:8px 0; font-weight:650; }}
    .error {{ color:#9b1c31; font-weight:700; }}
    @media (max-width:900px) {{
      h1 {{ font-size:26px; }}
      .summary, .model-grid, .role-head, .prov-grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>台灣女生 Clone TTS 多角色試聽 v3</h1>
    {offline_note}
    <p class="lead">這輪只看 F5-TTS、GPT-SoVITS v2、CosyVoice2。每個模型用同一批 Qwen3 台灣女生 reference，生成 6 個角色 × 4 句更台灣口吻句子。文字刻意用「欸、啦、喔、耶、慢慢講」測台灣感、尾音、穩定度。</p>
    <div class="summary">{''.join(summary_cards)}</div>
    {provenance}
  </header>
  <main>{''.join(role_sections)}</main>
</body>
</html>
"""


def main() -> int:
    INDEX.write_text(render(standalone=False), encoding="utf-8")
    STANDALONE.write_text(render(standalone=True), encoding="utf-8")
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    ok = [r for r in rows if r.get("output") and Path(str(r["output"])).exists()]
    print(INDEX)
    print(STANDALONE)
    print(f"rows={len(rows)} audio={len(ok)} standalone_mb={STANDALONE.stat().st_size / 1024 / 1024:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
