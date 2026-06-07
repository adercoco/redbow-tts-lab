#!/usr/bin/env python3
"""Build phone-readable Cosy vs ZipVoice vs distilled report."""

from __future__ import annotations

import base64
import collections
import html
import json
import mimetypes
import os
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
OUT = BASE / "cosy_zipvoice_distill_v1"
REPORT = OUT / "index.html"
STANDALONE = OUT / "cosy_vs_zipvoice_vs_distill_standalone.html"
REFS = OUT / "references" / "manifest.json"
COSY = OUT / "cosy_sweep_results.json"
ZIP = OUT / "zipvoice_compare_results.json"

KEEP_PACKS = ["raw_best2_7s", "clear_best2_7s"]
MODEL_ORDER = ["CosyVoice2 tuned teacher", "ZipVoice original 16-step", "ZipVoice-Distill stage2 4-step"]


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt_sec(value: object) -> str:
    try:
        return f"{float(value):.2f}s"
    except Exception:
        return "-"


def fmt_score(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "-"


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def src(path_value: str, *, standalone: bool) -> str:
    path = Path(path_value)
    if not path.exists():
        return ""
    if standalone:
        return data_uri(path)
    return esc(os.path.relpath(path, OUT))


def audio(path_value: str, *, standalone: bool) -> str:
    value = src(path_value, standalone=standalone)
    if not value:
        return '<div class="missing">沒有音檔</div>'
    return f'<audio controls preload="none" src="{value}"></audio>'


def load_rows() -> tuple[list[dict], list[dict], list[dict]]:
    return (
        json.loads(REFS.read_text(encoding="utf-8")),
        json.loads(COSY.read_text(encoding="utf-8")),
        json.loads(ZIP.read_text(encoding="utf-8")),
    )


def summary(rows: list[dict], family: str, pack: str | None = None) -> dict:
    subset = [row for row in rows if row.get("family") == family and row.get("status") == "ok"]
    if pack:
        subset = [row for row in subset if row.get("pack_id") == pack]
    secs = [float(row.get("seconds") or 0) for row in subset]
    scores = [float(row["speaker_cosine"]) for row in subset if row.get("speaker_cosine") is not None]
    rss = [float(row["peak_rss_mb_after_generate"]) for row in subset if row.get("peak_rss_mb_after_generate")]
    return {
        "n": len(subset),
        "sec": statistics.mean(secs) if secs else None,
        "score": statistics.mean(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "rss": max(rss) if rss else None,
        "model_size": next((row.get("model_size") for row in subset if row.get("model_size")), "-"),
    }


def render_ref_cards(refs: list[dict], *, standalone: bool) -> str:
    cards = []
    for ref in refs:
        if ref["pack_id"] not in ["raw_best2_7s", "clear_best2_7s", "clear_best3_11s"]:
            continue
        cards.append(
            f"""
            <article class="sample">
              <div class="sample-head"><b>{esc(ref['pack_id'])}</b><span>{esc(ref['duration'])}s / RMS {esc(ref['pack_stats']['rms_db'])} dB</span></div>
              <p>{esc(ref['note'])}</p>
              <p class="small">{esc(ref['ref_text'])}</p>
              {audio(ref['pack_audio'], standalone=standalone)}
            </article>
            """
        )
    return "".join(cards)


def model_card(name: str, stats: dict, note: str) -> str:
    return f"""
      <section class="card model">
        <h3>{esc(name)}</h3>
        <div class="bar"><span style="width:{max(8, min(100, float(stats.get('score') or 0) * 100)):.0f}%"></span></div>
        <div class="kv">
          <div><span>平均生成</span><b>{fmt_sec(stats['sec'])}</b></div>
          <div><span>相似分數</span><b>{fmt_score(stats['score'])}</b></div>
          <div><span>最高分</span><b>{fmt_score(stats['max_score'])}</b></div>
          <div><span>模型大小</span><b>{esc(stats['model_size'])}</b></div>
          <div><span>RSS</span><b>{'-' if stats['rss'] is None else f"{stats['rss']:.0f}MB"}</b></div>
        </div>
        <p>{esc(note)}</p>
      </section>
    """


def render_line_group(cosy_rows: list[dict], zip_rows: list[dict], pack: str, *, standalone: bool) -> str:
    cosy_by_text = {(row["pack_id"], row["text_id"]): row for row in cosy_rows if row.get("status") == "ok"}
    zip_by = {(row["family"], row["pack_id"], row["text_id"]): row for row in zip_rows if row.get("status") == "ok"}
    text_ids = ["line_01", "line_02", "line_03", "line_04"]
    blocks = []
    for text_id in text_ids:
        cosy = cosy_by_text[(pack, text_id)]
        cells = [
            f"""
            <article class="sample teacher">
              <div class="sample-head"><b>CosyVoice2 老師</b><span>{fmt_sec(cosy['seconds'])} / ref score {fmt_score(cosy.get('speaker_cosine'))}</span></div>
              {audio(cosy['output'], standalone=standalone)}
            </article>
            """
        ]
        for family in ["ZipVoice original 16-step", "ZipVoice-Distill stage2 4-step"]:
            row = zip_by[(family, pack, text_id)]
            cells.append(
                f"""
                <article class="sample">
                  <div class="sample-head"><b>{esc(family)}</b><span>{fmt_sec(row['seconds'])} / score {fmt_score(row.get('speaker_cosine'))}</span></div>
                  {audio(row['output'], standalone=standalone)}
                </article>
                """
            )
        blocks.append(
            f"""
            <section class="line">
              <p class="line-text">{esc(cosy['text'])}</p>
              <div class="sample-grid">{''.join(cells)}</div>
            </section>
            """
        )
    return "".join(blocks)


def render(*, standalone: bool) -> str:
    refs, cosy_rows, zip_rows = load_rows()
    all_model_cards = []
    cosy_raw = summary(cosy_rows, "CosyVoice2 tuned teacher", "raw_best2_7s")
    cosy_clear = summary(cosy_rows, "CosyVoice2 tuned teacher", "clear_best2_7s")
    zip_raw = summary(zip_rows, "ZipVoice original 16-step", "raw_best2_7s")
    distill_raw = summary(zip_rows, "ZipVoice-Distill stage2 4-step", "raw_best2_7s")
    all_model_cards.append(model_card("CosyVoice2 raw best2", cosy_raw, "聲紋分數最高；沒有額外放大 reference，但輸出已做安全 loudness normalize。"))
    all_model_cards.append(model_card("CosyVoice2 clear best2", cosy_clear, "reference 放大清晰版；比較清楚大聲，但 speaker score 略低。"))
    all_model_cards.append(model_card("ZipVoice original 16-step", zip_raw, "用 Cosy 老師音檔做 zero-shot reference，像度高但速度慢。"))
    all_model_cards.append(model_card("ZipVoice-Distill 4-step", distill_raw, "速度非常快，但目前不是 Cosy teacher 重訓版，像度明顯掉。"))

    raw_lines = render_line_group(cosy_rows, zip_rows, "raw_best2_7s", standalone=standalone)
    clear_lines = render_line_group(cosy_rows, zip_rows, "clear_best2_7s", standalone=standalone)
    refs_html = render_ref_cards(refs, standalone=standalone)
    mode = "Standalone HTML：音檔已嵌入，可手機離線開" if standalone else "Local HTML：音檔走相對路徑"

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cosy vs ZipVoice vs Distill 報告</title>
  <style>
    :root {{
      --bg:#f5f1ea; --paper:#fffdf8; --ink:#22201d; --muted:#756d64;
      --line:#ded5ca; --accent:#b13f3f; --accent2:#2f626a; --soft:#eee5da;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin:0; background:var(--bg); color:var(--ink);
      font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",Arial,sans-serif;
      line-height:1.58; letter-spacing:0;
    }}
    main {{ width:min(100%,780px); margin:0 auto; padding:18px 13px 48px; }}
    header {{ padding:20px 2px 16px; border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--accent); font-size:13px; font-weight:800; margin-bottom:8px; }}
    h1 {{ margin:0; font-size:27px; line-height:1.2; }}
    h2 {{ margin:26px 0 10px; font-size:22px; line-height:1.25; }}
    h3 {{ margin:0 0 10px; font-size:18px; }}
    p {{ margin:8px 0; font-size:16px; }}
    .lead {{ color:var(--muted); font-size:17px; }}
    .card,.section,.sample,.line {{
      background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:13px; margin-top:12px;
    }}
    .verdict {{ border-left:4px solid var(--accent); background:#fff8f0; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }}
    .bar {{ height:8px; border-radius:999px; background:var(--soft); overflow:hidden; }}
    .bar span {{ display:block; height:100%; background:linear-gradient(90deg,var(--accent),var(--accent2)); }}
    .kv {{ display:grid; gap:8px; margin:10px 0; }}
    .kv div {{ display:flex; justify-content:space-between; gap:14px; border-top:1px solid var(--soft); padding-top:8px; }}
    .kv span,.small,.sample-head span {{ color:var(--muted); font-size:13px; }}
    .kv b {{ font-size:14px; text-align:right; }}
    .sample-grid {{ display:grid; grid-template-columns:1fr; gap:10px; margin-top:10px; }}
    .sample {{ background:#fffaf3; }}
    .teacher {{ border-color:#d5b9a3; }}
    .sample-head {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }}
    .sample-head b {{ font-size:15px; }}
    .sample-head span {{ text-align:right; max-width:56%; overflow-wrap:anywhere; }}
    .line-text {{ font-weight:720; }}
    audio {{ width:100%; height:40px; margin-top:8px; }}
    details summary {{ cursor:pointer; font-weight:820; font-size:18px; }}
    @media (max-width:520px) {{
      main {{ padding:16px 12px 42px; }}
      h1 {{ font-size:25px; }}
      .grid {{ grid-template-columns:1fr; }}
      .sample-head {{ display:block; }}
      .sample-head span {{ display:block; max-width:100%; text-align:left; margin-top:4px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">cosy_zipvoice_distill_v1 / {esc(mode)} / 2026-06-04</div>
    <h1>Cosy vs ZipVoice vs 蒸餾後：女聲 clone 報告</h1>
    <p class="lead">目標：繼續刁 Cosy 的台灣溫柔女聲，測試 reference 放大清晰，然後看 ZipVoice 原始與 ZipVoice-Distill 是否能模仿 Cosy 老師。</p>
  </header>

  <section class="card verdict">
    <h2>目前判斷</h2>
    <p><b>Cosy raw best2 仍是目前最穩 teacher。</b> reference 放大清晰後，聲音更大更亮，但 speaker score 沒有提升。</p>
    <p><b>ZipVoice original 16-step 可以追 Cosy 聲音，但速度慢。</b> 同句平均約 {fmt_sec(zip_raw['sec'])}，相似分數約 {fmt_score(zip_raw['score'])}。</p>
    <p><b>ZipVoice-Distill 4-step 很快，但還不像。</b> 平均約 {fmt_sec(distill_raw['sec'])}，但相似分數約 {fmt_score(distill_raw['score'])}。要讓蒸餾後更像 Cosy，需要用 Cosy teacher 產 corpus 後重訓，不是只換 reference。</p>
  </section>

  <section>
    <h2>模型比較</h2>
    <div class="grid">{''.join(all_model_cards)}</div>
  </section>

  <details class="section" open>
    <summary>Reference 放大清晰測試</summary>
    {refs_html}
  </details>

  <section class="section">
    <h2>做法說明</h2>
    <p>先從乾淨女聲片段建立 reference pack。clear 版本做了 75Hz 高通、8.6kHz 低通、3.2kHz presence 微增益、RMS 放大到約 -21dB，並限制 peak 避免爆音。</p>
    <p>CosyVoice2 用這些 reference 產同 4 句。ZipVoice original 與 ZipVoice-Distill 再用 Cosy 的第一句輸出當 reference，生成同 4 句。分數是 CAMPPlus speaker cosine。</p>
    <p>重要限制：這份的「蒸餾後」是現有 ZipVoice-Distill stage2 4-step 權重，不是新做的 Cosy teacher 蒸餾。它能測速度與即時性，但不能代表 Cosy 聲音已經被完整蒸餾進模型。</p>
  </section>

  <section>
    <h2>Raw best2 三欄試聽</h2>
    {raw_lines}
  </section>

  <section>
    <h2>Clear best2 三欄試聽</h2>
    {clear_lines}
  </section>
</main>
</body>
</html>
"""


def main() -> int:
    REPORT.write_text(render(standalone=False), encoding="utf-8")
    STANDALONE.write_text(render(standalone=True), encoding="utf-8")
    print(REPORT)
    print(STANDALONE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
