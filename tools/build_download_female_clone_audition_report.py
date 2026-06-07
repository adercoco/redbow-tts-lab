#!/usr/bin/env python3
"""Build a phone-readable clone audition report for Downloads female voice refs."""

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
OUT = BASE / "clone_audition_v1"
RESULTS = OUT / "clone_audition_results.json"

MODEL_META = {
    "GPT-SoVITS v2": {
        "size": "1.2GB pretrained models / 1.9GB repo",
        "role": "標準 zero-shot clone baseline；reference 必須 3-10 秒。",
        "path": ROOT / "external" / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models",
    },
    "CosyVoice2-0.5B": {
        "size": "4.5GB local pretrained folder",
        "role": "較大的高品質 zero-shot clone baseline，聲音自然度通常強。",
        "path": ROOT / "external" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B",
    },
    "Qwen3 1.7B VoiceDesign ref attempt": {
        "size": "2.2GB local 4bit HF cache",
        "role": "VoiceDesign + reference 嘗試；速度快，但不是純 clone API。",
        "path": Path("~/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit").expanduser(),
    },
    "F5-TTS v1 Base": {
        "size": "1.3GB local HF cache",
        "role": "F5-TTS v1 Base 標準 clone baseline；這次每句重啟 CLI，所以秒數含載入成本。",
        "path": Path("~/.cache/huggingface/hub/models--SWivid--F5-TTS").expanduser(),
    },
}

FAMILY_ORDER = [
    "GPT-SoVITS v2",
    "CosyVoice2-0.5B",
    "Qwen3 1.7B VoiceDesign ref attempt",
    "F5-TTS v1 Base",
]

STYLE_LABELS = {
    "tw_soft": "台灣口吻：先別急",
    "tw_calm": "台灣口吻：慢慢說",
}


def escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt_seconds(value: object) -> str:
    try:
        return f"{float(value):.2f}s"
    except Exception:
        return "-"


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def rel_or_embed(path_value: str, *, standalone: bool) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists():
        return ""
    if standalone:
        return data_uri(path)
    return escape(os.path.relpath(path, OUT))


def summarize(rows: list[dict]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row.get("family") == family]
        ok = [float(row["seconds"]) for row in family_rows if row.get("status") == "ok"]
        scored = [float(row["speaker_cosine"]) for row in family_rows if row.get("speaker_cosine") is not None]
        failed = [row for row in family_rows if row.get("status") != "ok"]
        summary[family] = {
            "total": len(family_rows),
            "ok": len(ok),
            "failed": len(failed),
            "avg": statistics.mean(ok) if ok else None,
            "min": min(ok) if ok else None,
            "max": max(ok) if ok else None,
            "speaker_avg": statistics.mean(scored) if scored else None,
            "speaker_max": max(scored) if scored else None,
        }
    return summary


def render_audio(src: str) -> str:
    if not src:
        return '<div class="missing">沒有輸出</div>'
    return f'<audio controls preload="none" src="{src}"></audio>'


def render_report(*, standalone: bool) -> str:
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    by_ref: dict[str, list[dict]] = collections.defaultdict(list)
    refs: dict[str, dict] = {}
    for row in rows:
        by_ref[row["ref_id"]].append(row)
        refs.setdefault(row["ref_id"], row)
    summary = summarize(rows)

    total_ok = sum(item["ok"] for item in summary.values())
    total_failed = sum(item["failed"] for item in summary.values())
    ref_count = len(refs)

    summary_cards = []
    for family in FAMILY_ORDER:
        item = summary[family]
        meta = MODEL_META[family]
        avg = "-" if item["avg"] is None else f"{item['avg']:.2f}s"
        minmax = "-" if item["min"] is None else f"{item['min']:.2f}-{item['max']:.2f}s"
        speaker = "-" if item["speaker_avg"] is None else f"{item['speaker_avg']:.3f} avg / {item['speaker_max']:.3f} max"
        summary_cards.append(
            f"""
            <section class="model-card">
              <div class="model-title">{escape(family)}</div>
              <div class="stat-row"><span>成功</span><b>{item['ok']} / {item['total']}</b></div>
              <div class="stat-row"><span>平均生成</span><b>{avg}</b></div>
              <div class="stat-row"><span>範圍</span><b>{minmax}</b></div>
              <div class="stat-row"><span>聲紋相似</span><b>{speaker}</b></div>
              <div class="stat-row"><span>模型大小</span><b>{escape(meta['size'])}</b></div>
              <p>{escape(meta['role'])}</p>
            </section>
            """
        )

    ref_sections = []
    for ref_id in sorted(refs):
        ref = refs[ref_id]
        ref_src = rel_or_embed(ref.get("ref_audio", ""), standalone=standalone)
        ref_duration = fmt_seconds(ref.get("ref_duration"))
        ref_f0 = "-"
        try:
            ref_f0 = f"{float(ref.get('ref_median_f0')):.0f}Hz"
        except Exception:
            pass
        model_blocks = []
        for family in FAMILY_ORDER:
            family_rows = [row for row in by_ref[ref_id] if row.get("family") == family]
            chips = []
            for row in sorted(family_rows, key=lambda item: item.get("text_id", "")):
                style = STYLE_LABELS.get(row.get("text_id"), row.get("text_id", ""))
                if row.get("status") == "ok":
                    src = rel_or_embed(row.get("output", ""), standalone=standalone)
                    body = render_audio(src)
                    score = row.get("speaker_cosine")
                    score_text = "" if score is None else f" / score {float(score):.3f}"
                    status = f"{fmt_seconds(row.get('seconds'))}{score_text}"
                else:
                    body = f'<div class="error">{escape(row.get("error"))}</div>'
                    status = "failed"
                chips.append(
                    f"""
                    <div class="sample">
                      <div class="sample-head"><span>{escape(style)}</span><b>{escape(status)}</b></div>
                      <div class="sample-text">{escape(row.get('text'))}</div>
                      {body}
                    </div>
                    """
                )
            if not chips:
                chips.append('<div class="missing">沒有這個模型的輸出</div>')
            model_blocks.append(
                f"""
                <section class="family">
                  <h3>{escape(family)}</h3>
                  {''.join(chips)}
                </section>
                """
            )

        ref_sections.append(
            f"""
            <details class="ref" open>
              <summary>
                <span>{escape(ref_id)}</span>
                <b>{ref_duration} / {ref_f0}</b>
              </summary>
              <div class="ref-inner">
                <div class="ref-audio">
                  <div class="label">Reference 原音</div>
                  {render_audio(ref_src)}
                </div>
                <div class="transcript">
                  <div class="label">Whisper 逐字稿草稿</div>
                  <p>{escape(ref.get('ref_text'))}</p>
                </div>
                {''.join(model_blocks)}
              </div>
            </details>
            """
        )

    mode = "Standalone HTML：音檔已嵌入，可離線分享" if standalone else "Local HTML：音檔用相對路徑載入"
    html_doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>女聲 Reference Clone Audition v1</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #fbfaf7;
      --ink: #1f2428;
      --muted: #687076;
      --line: #ddd8cf;
      --panel: #ffffff;
      --accent: #b3263a;
      --soft: #f3ebe5;
      --ok: #28704a;
      --bad: #9f2f24;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      line-height: 1.58;
      letter-spacing: 0;
    }}
    main {{
      width: min(100%, 980px);
      margin: 0 auto;
      padding: 20px 16px 44px;
    }}
    .hero {{
      padding: 22px 0 16px;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 7vw, 46px);
      line-height: 1.08;
      letter-spacing: 0;
    }}
    .hero p, .note p, .model-card p {{
      margin: 0;
      color: var(--muted);
      font-size: 16px;
    }}
    .quick {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin: 18px 0;
    }}
    .quick div, .note, .model-card, .family, .transcript, .ref-audio {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .quick div {{
      padding: 12px;
      min-height: 78px;
    }}
    .quick span, .label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .quick b {{
      display: block;
      font-size: 20px;
      line-height: 1.15;
    }}
    .note {{
      padding: 14px;
      margin: 14px 0 20px;
      background: var(--soft);
    }}
    .models {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin: 14px 0 22px;
    }}
    .model-card {{
      padding: 14px;
    }}
    .model-title {{
      font-size: 18px;
      font-weight: 800;
      margin-bottom: 10px;
    }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 6px 0;
      border-top: 1px solid #eee9e2;
      font-size: 14px;
    }}
    .stat-row span {{ color: var(--muted); }}
    .stat-row b {{ text-align: right; }}
    h2 {{
      margin: 24px 0 12px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .ref {{
      margin: 14px 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: clip;
    }}
    summary {{
      cursor: pointer;
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px;
      background: #fff8f6;
      border-bottom: 1px solid var(--line);
      font-size: 18px;
      font-weight: 800;
    }}
    summary::-webkit-details-marker {{ display: none; }}
    summary b {{
      color: var(--muted);
      font-size: 14px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .ref-inner {{
      padding: 12px;
      display: grid;
      gap: 12px;
    }}
    .ref-audio, .transcript, .family {{
      padding: 12px;
    }}
    .transcript p {{
      margin: 0;
      font-size: 16px;
    }}
    .family h3 {{
      margin: 0 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .sample {{
      padding: 10px 0 12px;
      border-top: 1px solid #eee9e2;
    }}
    .sample:first-of-type {{
      border-top: 0;
      padding-top: 0;
    }}
    .sample-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
      font-size: 14px;
      color: var(--muted);
    }}
    .sample-head b {{
      color: var(--ok);
      white-space: nowrap;
    }}
    .sample-text {{
      font-size: 15px;
      margin-bottom: 8px;
    }}
    audio {{
      width: 100%;
      height: 42px;
      display: block;
    }}
    .missing, .error {{
      padding: 10px;
      border-radius: 8px;
      font-size: 14px;
    }}
    .missing {{
      color: var(--muted);
      background: #f4f2ed;
    }}
    .error {{
      color: var(--bad);
      background: #fff1ed;
      border: 1px solid #f2c6bc;
    }}
    @media (max-width: 720px) {{
      main {{ padding: 16px 12px 36px; }}
      .quick {{ grid-template-columns: repeat(2, 1fr); }}
      .models {{ grid-template-columns: 1fr; }}
      .hero p, .note p {{ font-size: 15px; }}
      summary {{ font-size: 17px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="eyebrow">{escape(mode)}・2026-06-04</div>
      <h1>女聲 Clone Audition v1</h1>
      <p>從下載資料夾音檔中切出乾淨女聲片段，補 Whisper 逐字稿草稿，再餵給 F5-TTS、GPT-SoVITS、CosyVoice2、Qwen3 1.7B 做同句試聽。</p>
    </section>
    <section class="quick">
      <div><span>Reference</span><b>{ref_count} 段</b></div>
      <div><span>成功輸出</span><b>{total_ok}</b></div>
      <div><span>失敗輸出</span><b>{total_failed}</b></div>
      <div><span>清理後女聲</span><b>40.0s</b></div>
    </section>
    <section class="note">
      <p>逐字稿是 mlx-community/whisper-large-v3-mlx 自動辨識草稿。這次先拿來 audition 聽音色和 clone 傾向；正式訓練前應手工校正逐字稿，並優先挑 3-10 秒、沒有男聲、沒有配樂、沒有唱歌的 reference。聲紋相似分數是 CosyVoice CAMPPlus embedding cosine，越接近 1 越貼 reference，只能當初篩。</p>
    </section>
    <h2>模型總覽</h2>
    <section class="models">
      {''.join(summary_cards)}
    </section>
    <section class="note">
      <p>GPT-SoVITS v2 失敗的 12 筆不是推論崩壞，而是 reference 太短，低於它要求的 3 秒門檻。F5 這次用 CLI 逐句重啟，所以秒數含載入成本；做成 app 或服務時應該常駐模型，只比較「已載入後生成」才公平。</p>
    </section>
    <h2>逐段試聽</h2>
    {''.join(ref_sections)}
  </main>
</body>
</html>
"""
    return html_doc


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index = OUT / "index.html"
    standalone = OUT / "clone_audition_v1_standalone.html"
    index.write_text(render_report(standalone=False), encoding="utf-8")
    standalone.write_text(render_report(standalone=True), encoding="utf-8")
    print(index)
    print(standalone)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
