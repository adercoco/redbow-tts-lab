#!/usr/bin/env python3
"""Build the complete large-model clone/fine-tune audition report."""

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
OUT = BASE / "clone_audition_v2_large_models"
RESULTS = OUT / "all_results_scored.json"
V1_RESULTS = BASE / "clone_audition_v1" / "clone_audition_results.json"
REPORT = OUT / "index.html"
STANDALONE = OUT / "clone_audition_v2_large_models_standalone.html"

MODEL_ORDER = [
    "CosyVoice2-0.5B pack",
    "F5-TTS v1 Base pack",
    "GPT-SoVITS v2 aux refs",
    "IndexTTS2 pack",
    "F5-TTS fine-tuned 40 updates",
]

MODEL_META = {
    "CosyVoice2-0.5B pack": {
        "size": "4.5GB local pretrained folder",
        "memory": "未做獨立 peak RSS；CPU 推論實測可跑，實際 app 端仍過大",
        "license": "Apache-2.0 路線較適合商用驗證",
        "role": "zero-shot clone。這輪用 7 秒 / 11 秒 reference pack，比單句 reference 更穩。",
    },
    "F5-TTS v1 Base pack": {
        "size": "1.3GB HF cache",
        "memory": "未做獨立 peak RSS；CLI 每句重啟，秒數含載入成本",
        "license": "MIT",
        "role": "zero-shot clone。Flow-matching 架構，聲紋分數接近 Cosy，但速度慢。",
    },
    "GPT-SoVITS v2 aux refs": {
        "size": "1.2GB pretrained models",
        "memory": "未做獨立 peak RSS；CPU 跑 aux reference 較慢",
        "license": "MIT",
        "role": "zero-shot clone + auxiliary references。這輪用 1 主 ref + 5 aux refs。",
    },
    "IndexTTS2 pack": {
        "size": "8.3GB checkpoints folder",
        "memory": "未做獨立 peak RSS；MPS 熱機後約 8-12 秒/句，首句含載入 47.9 秒",
        "license": "LicenseRef-Bilibili-IndexTTS；商用需另外確認，不適合直接當商用蒸餾老師",
        "role": "較大的高品質 clone 候選。技術上可跑，但授權和體積都不適合手機 app runtime。",
    },
    "F5-TTS fine-tuned 40 updates": {
        "size": "5.0GB active model_last.pt；checkpoint folder 16GB",
        "memory": "未做獨立 peak RSS；目前每句 CLI 載入 5GB checkpoint，約 46.8 秒/句",
        "license": "MIT base；微調資料授權依你的原始音檔授權",
        "role": "完整微調 proof-of-work：12 段、約 40 秒資料、40 updates。這不是可部署小模型。",
    },
}

TEXT_LABELS = {
    "tw_soft": "台灣口吻：先別急",
    "tw_calm": "台灣口吻：慢慢說",
    "tw_decide": "台灣口吻：慢慢決定",
    "tw_confirm": "台灣口吻：一起確認",
}


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


def audio_src(path_value: str, *, standalone: bool) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists():
        return ""
    if standalone:
        return data_uri(path)
    return esc(os.path.relpath(path, OUT))


def audio_tag(path_value: str, *, standalone: bool) -> str:
    src = audio_src(path_value, standalone=standalone)
    if not src:
        return '<div class="missing">沒有音檔</div>'
    return f'<audio controls preload="none" src="{src}"></audio>'


def summarize(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for family in MODEL_ORDER:
        family_rows = [row for row in rows if row.get("family") == family and row.get("status") == "ok"]
        seconds = [float(row["seconds"]) for row in family_rows]
        scores = [float(row["speaker_cosine"]) for row in family_rows if row.get("speaker_cosine") is not None]
        hot_seconds = seconds
        if family == "IndexTTS2 pack" and len(seconds) > 1:
            hot_seconds = [sec for sec in seconds if sec < 30]
        out[family] = {
            "n": len(family_rows),
            "avg_sec": statistics.mean(seconds) if seconds else None,
            "avg_hot_sec": statistics.mean(hot_seconds) if hot_seconds else None,
            "min_sec": min(seconds) if seconds else None,
            "max_sec": max(seconds) if seconds else None,
            "avg_score": statistics.mean(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "min_score": min(scores) if scores else None,
        }
    return out


def summarize_v1_qwen() -> dict[str, object] | None:
    if not V1_RESULTS.exists():
        return None
    rows = json.loads(V1_RESULTS.read_text(encoding="utf-8"))
    qwen = [row for row in rows if row.get("family") == "Qwen3 1.7B VoiceDesign ref attempt" and row.get("status") == "ok"]
    if not qwen:
        return None
    seconds = [float(row["seconds"]) for row in qwen]
    scores = [float(row.get("speaker_cosine", 0)) for row in qwen if row.get("speaker_cosine") is not None]
    examples = qwen[:4]
    return {
        "n": len(qwen),
        "avg_sec": statistics.mean(seconds),
        "avg_score": statistics.mean(scores) if scores else None,
        "examples": examples,
    }


def summary_card(family: str, stats: dict) -> str:
    meta = MODEL_META[family]
    hot = ""
    if family == "IndexTTS2 pack":
        hot = f"<div><span>熱機後平均</span><b>{fmt_sec(stats['avg_hot_sec'])}</b></div>"
    return f"""
      <section class="card model">
        <h3>{esc(family)}</h3>
        <div class="meter"><span style="width:{max(8, min(100, float(stats.get('avg_score') or 0) * 100)):.0f}%"></span></div>
        <div class="kv"><div><span>成功音檔</span><b>{stats['n']}</b></div><div><span>平均生成</span><b>{fmt_sec(stats['avg_sec'])}</b></div>{hot}<div><span>相似分數</span><b>{fmt_score(stats['avg_score'])}</b></div><div><span>模型大小</span><b>{esc(meta['size'])}</b></div></div>
        <p>{esc(meta['role'])}</p>
        <p class="small">{esc(meta['memory'])}</p>
        <p class="small">{esc(meta['license'])}</p>
      </section>
    """


def render_samples(rows: list[dict], *, standalone: bool) -> str:
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            by_family[row["family"]].append(row)
    parts: list[str] = []
    for family in MODEL_ORDER:
        family_rows = sorted(by_family.get(family, []), key=lambda r: (r.get("pack_id", ""), r.get("text_id", "")))
        if not family_rows:
            continue
        cards = []
        for row in family_rows:
            cards.append(
                f"""
                <article class="sample">
                  <div class="sample-head">
                    <b>{esc(TEXT_LABELS.get(row.get('text_id'), row.get('text_id')))}</b>
                    <span>{esc(row.get('pack_id'))} / {fmt_sec(row.get('seconds'))} / score {fmt_score(row.get('speaker_cosine'))}</span>
                  </div>
                  <p>{esc(row.get('text'))}</p>
                  {audio_tag(row.get('output', ''), standalone=standalone)}
                </article>
                """
            )
        parts.append(
            f"""
            <details class="section" open>
              <summary>{esc(family)}</summary>
              <div class="sample-list">{''.join(cards)}</div>
            </details>
            """
        )
    return "\n".join(parts)


def render_references(rows: list[dict], *, standalone: bool) -> str:
    refs: dict[str, dict] = {}
    for row in rows:
        if row.get("ref_id") and row.get("ref_audio"):
            refs.setdefault(row["ref_id"], row)
    items = []
    for ref_id, row in sorted(refs.items()):
        files = ", ".join(row.get("ref_files", [])) or "main + aux clips"
        items.append(
            f"""
            <article class="sample">
              <div class="sample-head"><b>{esc(ref_id)}</b><span>{esc(files)}</span></div>
              <p>{esc(row.get('ref_text'))}</p>
              {audio_tag(row.get('ref_audio', ''), standalone=standalone)}
            </article>
            """
        )
    return "".join(items)


def render_qwen(*, standalone: bool) -> str:
    qwen = summarize_v1_qwen()
    if not qwen:
        return ""
    examples = []
    for row in qwen["examples"]:
        examples.append(
            f"""
            <article class="sample">
              <div class="sample-head"><b>{esc(row.get('ref_id'))}</b><span>{fmt_sec(row.get('seconds'))} / score {fmt_score(row.get('speaker_cosine'))}</span></div>
              <p>{esc(row.get('text'))}</p>
              {audio_tag(row.get('output', ''), standalone=standalone)}
            </article>
            """
        )
    return f"""
      <section class="section">
        <h2>Qwen 對照</h2>
        <p>Qwen3 1.7B VoiceDesign 4bit 有喂 reference audio/text 做過 24 段嘗試，平均 {fmt_sec(qwen['avg_sec'])}，但 speaker cosine 平均只有 {fmt_score(qwen['avg_score'])}。結論：它適合當 VoiceDesign 老師，不適合用這段音檔做純 clone。</p>
        <div class="sample-list">{''.join(examples)}</div>
      </section>
    """


def render(*, standalone: bool) -> str:
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    stats = summarize(rows)
    cards = "\n".join(summary_card(family, stats[family]) for family in MODEL_ORDER)
    sample_rows = rows
    ref_html = render_references(rows, standalone=standalone)
    sample_html = render_samples(sample_rows, standalone=standalone)
    qwen_html = render_qwen(standalone=standalone)
    mode = "Standalone：音檔已嵌入，可丟到手機離線開" if standalone else "Local：音檔用相對路徑載入"

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>台灣溫柔女聲 Clone / 微調完整報告 v2</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f2ea;
      --paper: #fffdf8;
      --ink: #24211d;
      --muted: #786f65;
      --line: #ded4c7;
      --accent: #b34a4a;
      --accent2: #315f68;
      --soft: #efe6dc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", Arial, sans-serif;
      line-height: 1.58;
      letter-spacing: 0;
    }}
    main {{
      width: min(100%, 760px);
      margin: 0 auto;
      padding: 20px 14px 48px;
    }}
    header {{
      padding: 22px 2px 16px;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.2;
      font-weight: 820;
    }}
    h2 {{
      margin: 28px 0 10px;
      font-size: 22px;
      line-height: 1.25;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 18px;
      line-height: 1.25;
    }}
    p {{
      margin: 8px 0;
      font-size: 16px;
    }}
    .lead {{
      color: var(--muted);
      font-size: 17px;
      margin-top: 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .card, .section, details.section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-top: 12px;
    }}
    .verdict {{
      border-left: 4px solid var(--accent);
      background: #fff9f3;
    }}
    .kv {{
      display: grid;
      gap: 8px;
      margin: 10px 0;
    }}
    .kv div {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      border-top: 1px solid var(--soft);
      padding-top: 8px;
    }}
    .kv span, .small, .sample-head span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .kv b {{
      text-align: right;
      font-size: 14px;
    }}
    .meter {{
      height: 8px;
      background: var(--soft);
      border-radius: 999px;
      overflow: hidden;
    }}
    .meter span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
    }}
    summary {{
      cursor: pointer;
      font-weight: 800;
      font-size: 18px;
      list-style-position: inside;
    }}
    .sample-list {{
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }}
    .sample {{
      border: 1px solid var(--soft);
      border-radius: 8px;
      padding: 12px;
      background: #fffaf2;
    }}
    .sample-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 8px;
    }}
    .sample-head b {{
      font-size: 15px;
    }}
    .sample-head span {{
      text-align: right;
      max-width: 55%;
      overflow-wrap: anywhere;
    }}
    audio {{
      width: 100%;
      height: 40px;
      margin-top: 8px;
    }}
    .steps {{
      counter-reset: step;
      display: grid;
      gap: 10px;
    }}
    .step {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .step b {{
      color: var(--accent2);
    }}
    code {{
      background: var(--soft);
      border-radius: 4px;
      padding: 1px 4px;
    }}
    @media (max-width: 520px) {{
      main {{ padding: 16px 12px 42px; }}
      h1 {{ font-size: 25px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .sample-head {{ display: block; }}
      .sample-head span {{ display: block; max-width: 100%; text-align: left; margin-top: 4px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Clone audition v2 / {esc(mode)} / 2026-06-04</div>
    <h1>台灣溫柔女聲：大模型 clone 與完整微調報告</h1>
    <p class="lead">資料來源是你授權的女聲音檔；我已切出乾淨女生片段、排除男聲、配樂、唱歌風險，補逐字稿，做 multi-reference zero-shot 與 F5 完整微調 proof-of-work。</p>
  </header>

  <section class="card verdict">
    <h2>目前結論</h2>
    <p><b>最值得繼續的是 CosyVoice2-0.5B pack_best2_7s。</b> 它在這輪的平均聲紋分數最高，平均生成約 6.7 秒/句，且比 11 秒 reference 更快也略高分。</p>
    <p><b>F5 完整微調已做，但目前不值得當主線。</b> 40 秒資料 + 40 updates 沒有明顯超過 F5 base，checkpoint 反而變 5GB，推論也因 CLI 載入變成約 47 秒/句。要做真正可部署小模型，需要更多乾淨資料與重新設計 student，而不是直接拿這個 checkpoint 上手機。</p>
    <p><b>IndexTTS2 聽感可列候選，但不適合直接商用蒸餾或手機 runtime。</b> 本地 checkpoint 8.3GB，license 明確要求商用合作確認，且限制用它改善商用 AI model。</p>
  </section>

  <section>
    <h2>模型並排比較</h2>
    <div class="grid">{cards}</div>
  </section>

  <section class="section">
    <h2>怎麼做的</h2>
    <div class="steps">
      <div class="step"><b>1. 清資料</b><p>從音檔切出候選片段，再過濾成 12 段乾淨女聲，總長約 40 秒。男聲、明顯配樂、唱歌片段不拿來當 reference 或微調資料。</p></div>
      <div class="step"><b>2. 補逐字稿</b><p>為每段做 TTS-ready transcript，因 F5 / GPT-SoVITS / CosyVoice 都需要 reference text 來對齊聲音和文字。</p></div>
      <div class="step"><b>3. 多 reference</b><p>建立 <code>pack_best2_7s</code> 和 <code>pack_best3_11s</code>。結果顯示不是越長越好：CosyVoice2 對 7 秒 pack 分數更高、速度更快；F5 用 11 秒 pack 也更慢。</p></div>
      <div class="step"><b>4. Zero-shot clone</b><p>同一句測試句餵給 CosyVoice2、F5-TTS、GPT-SoVITS、IndexTTS2，直接比較輸出聲音、秒數與 speaker cosine。</p></div>
      <div class="step"><b>5. 完整微調</b><p>用 12 段、約 40 秒資料對 F5-TTS v1 Base 做 40 updates 微調，輸出 <code>model_last.pt</code>。這證明流程可跑，但資料量太小，還不是能上手機的小模型。</p></div>
      <div class="step"><b>6. 評分</b><p>用 CosyVoice2 內附 CAMPPlus speaker embedding 做 reference-to-clone cosine。這是機器輔助排序，不取代實際聽感；GPT-SoVITS aux refs 用 reference pool 的最高分。</p></div>
    </div>
  </section>

  <section class="section">
    <h2>下一步技術判斷</h2>
    <p>如果目標是「中階手機上快速跑」，這些大模型都還不是 runtime。務實路線是：先用 CosyVoice2 或已確認授權的老師產出更乾淨、更大量的同聲線 corpus，再訓練 ZipVoice / Piper 類小 student，並以 ONNX int8 或 Core ML 做部署。</p>
    <p>如果目標是「最像這個授權女聲」，要補資料。40 秒只夠做 clone audition，不夠穩定微調；建議至少 10-30 分鐘乾淨語音，最好是無背景音、同麥克風、正常講話，並有準確逐字稿。</p>
  </section>

  {qwen_html}

  <details class="section">
    <summary>Reference packs</summary>
    <div class="sample-list">{ref_html}</div>
  </details>

  <section>
    <h2>試聽結果</h2>
    {sample_html}
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
