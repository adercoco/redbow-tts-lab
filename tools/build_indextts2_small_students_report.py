#!/usr/bin/env python3
"""Build a phone-readable report for IndexTTS2 -> Piper/Matcha smoke tests."""

from __future__ import annotations

import base64
import html
import json
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
REPORT_DIR = BASE / "reports" / "indextts2_to_piper_matcha_smoke_v1"
TEACHER = BASE / "teacher_indextts2_distill_smoke_v1"
PIPER = BASE / "students" / "piper_medium_indextts2_smoke100"
MATCHA = BASE / "students" / "matcha_pinyin_indextts2_smoke80"

ROWS = [
    {
        "text": "你先不要急，我们慢慢来，把事情一件一件处理好。",
        "teacher": TEACHER / "audio" / "indextts2_tw_0001.wav",
        "piper": PIPER / "samples_builtin" / "piper_builtin_1.wav",
        "matcha": MATCHA / "samples" / "matcha_1.wav",
    },
    {
        "text": "我刚刚看了一下，应该不是你的问题，你不用太担心。",
        "teacher": TEACHER / "audio" / "indextts2_tw_0002.wav",
        "piper": PIPER / "samples_builtin" / "piper_builtin_2.wav",
        "matcha": MATCHA / "samples" / "matcha_2.wav",
    },
    {
        "text": "没关系啦，你先讲，我在这边听，真的不用紧张。",
        "teacher": TEACHER / "audio" / "indextts2_tw_0006.wav",
        "piper": PIPER / "samples_builtin" / "piper_builtin_3.wav",
        "matcha": MATCHA / "samples" / "matcha_3.wav",
    },
]


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def audio_src(path: Path, standalone: bool) -> str:
    if standalone:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:audio/wav;base64,{data}"
    return html.escape(str(path.relative_to(REPORT_DIR)))


def rel_copy_assets() -> None:
    audio_dir = REPORT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for row in ROWS:
        for key in ("teacher", "piper", "matcha"):
            src = row[key]
            dst = audio_dir / f"{key}_{src.name}"
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            row[f"{key}_rel"] = dst


def render(standalone: bool) -> str:
    teacher_manifest = json.loads((TEACHER / "manifest.json").read_text(encoding="utf-8"))
    teacher_times = [r["seconds"] for r in teacher_manifest if r.get("status") == "ok"]
    teacher_avg = sum(teacher_times) / len(teacher_times)

    cards = []
    for i, row in enumerate(ROWS, start=1):
        cells = []
        for key, label, note in [
            ("teacher", "IndexTTS2 老師", "大模型 clone，這次當老師語料"),
            ("piper", "VITS/Piper 學生", "60MB ONNX，正確 Piper 中文前端"),
            ("matcha", "Matcha 學生", "拼音前端 + HiFi-GAN，80-step smoke"),
        ]:
            src_path = row[key] if standalone else row[f"{key}_rel"]
            cells.append(
                f"""
                <div class="voice">
                  <div class="voiceTop"><strong>{label}</strong><span>{note}</span></div>
                  <audio controls preload="metadata" src="{audio_src(src_path, standalone)}"></audio>
                  <div class="meta">音檔長度 {wav_seconds(row[key]):.2f}s</div>
                </div>
                """
            )
        cards.append(
            f"""
            <section class="sample">
              <div class="sampleNo">Sample {i}</div>
              <h2>{html.escape(row['text'])}</h2>
              <div class="voices">{''.join(cells)}</div>
            </section>
            """
        )

    piper_onnx = PIPER / "taiwan_indextts2_piper_medium_smoke100.onnx"
    matcha_ckpt = MATCHA / "checkpoints" / "last.ckpt"
    hifigan = ROOT / "models" / "matcha" / "hifigan_T2_v1"

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IndexTTS2 小模型學生 smoke report</title>
  <style>
    :root {{
      --bg: #f7f3ed;
      --ink: #202124;
      --muted: #66615a;
      --line: #ded4c7;
      --panel: #fffaf2;
      --red: #b4232a;
      --green: #356b4f;
      --blue: #335f8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Noto Sans TC", "PingFang TC", sans-serif;
      line-height: 1.58;
    }}
    main {{
      width: min(920px, 100%);
      margin: 0 auto;
      padding: 20px 14px 48px;
    }}
    .hero {{
      padding: 24px 4px 18px;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      color: var(--red);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0;
      margin-bottom: 8px;
    }}
    h1 {{
      font-size: clamp(28px, 7vw, 44px);
      line-height: 1.12;
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    .lead {{
      font-size: 17px;
      color: var(--muted);
      margin: 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin: 18px 0;
    }}
    .metric, .sample, .note {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric b {{
      display: block;
      font-size: 22px;
      line-height: 1.15;
      margin-bottom: 4px;
    }}
    .metric span, .meta, .voiceTop span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .sample {{
      margin: 14px 0;
    }}
    .sampleNo {{
      color: var(--blue);
      font-size: 13px;
      font-weight: 800;
      margin-bottom: 6px;
    }}
    h2 {{
      font-size: 20px;
      line-height: 1.35;
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    .voices {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }}
    .voice {{
      border-top: 3px solid var(--line);
      padding-top: 10px;
    }}
    .voice:nth-child(1) {{ border-color: var(--red); }}
    .voice:nth-child(2) {{ border-color: var(--green); }}
    .voice:nth-child(3) {{ border-color: var(--blue); }}
    .voiceTop strong {{
      display: block;
      font-size: 16px;
      margin-bottom: 2px;
    }}
    audio {{
      width: 100%;
      margin: 8px 0 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      display: table;
      margin: 14px 0;
    }}
    th, td {{
      padding: 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{ color: var(--muted); font-size: 12px; }}
    tr:last-child td {{ border-bottom: 0; }}
    .note {{
      margin: 14px 0;
      font-size: 15px;
    }}
    code {{
      font-size: 12px;
      word-break: break-all;
    }}
    @media (max-width: 720px) {{
      main {{ padding: 18px 12px 42px; }}
      .summary, .voices {{ grid-template-columns: 1fr; }}
      table {{ display: block; overflow-x: auto; }}
      th, td {{ min-width: 132px; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div class="eyebrow">IndexTTS2 → VITS/Piper / Matcha smoke・2026-06-05</div>
    <h1>兩條手機小模型學生先跑通</h1>
    <p class="lead">這份不是最終聲音，是第一輪「能不能用 IndexTTS2 當老師，讓小模型學生學聲音方向」的實測報告。</p>
  </section>

  <section class="summary">
    <div class="metric"><b>0.09-0.15s</b><span>Piper 載入後單句生成，CPU ONNX</span></div>
    <div class="metric"><b>0.52s</b><span>Matcha + HiFi-GAN 單句生成，CPU PyTorch</span></div>
    <div class="metric"><b>{teacher_avg:.1f}s</b><span>IndexTTS2 老師平均單句生成</span></div>
  </section>

  <div class="note">
    <strong>目前判斷：</strong>Piper/VITS 是最接近手機即時的路線；Matcha/FastSpeech-style 速度也有潛力，但這次只有 66 秒語料、80 step、拼音無聲調前端，所以聽感只能當架構 smoke。真正要像 IndexTTS2，需要擴語料和長訓。
  </div>

  {''.join(cards)}

  <h2>模型與資源對比</h2>
  <table>
    <thead><tr><th>項目</th><th>IndexTTS2 老師</th><th>VITS/Piper 學生</th><th>Matcha/FastSpeech-style 學生</th></tr></thead>
    <tbody>
      <tr><td>定位</td><td>高品質 clone 老師</td><td>小、快、可 ONNX 手機部署</td><td>非 AR acoustic model + vocoder，速度潛力好</td></tr>
      <tr><td>訓練資料</td><td>原 reference pack</td><td>16 句 IndexTTS2 teacher wav，約 66 秒</td><td>同 16 句，但文字轉無聲調拼音</td></tr>
      <tr><td>部署大小</td><td>本地 checkpoint 約 8.3GB</td><td>{size_mb(piper_onnx):.1f}MB ONNX</td><td>{size_mb(matcha_ckpt):.1f}MB checkpoint + {size_mb(hifigan):.1f}MB vocoder</td></tr>
      <tr><td>峰值記憶體</td><td>未重測；大模型不適合手機</td><td>約 280MB max RSS；含載入+生成</td><td>約 1.20GB max RSS；PyTorch + HiFi-GAN</td></tr>
      <tr><td>生成時間</td><td>本批平均 {teacher_avg:.1f}s / 句</td><td>載入後 0.09-0.15s / 句</td><td>載入後約 0.52s / 句</td></tr>
      <tr><td>目前風險</td><td>太大太慢</td><td>66 秒資料太少，音色可能還沒吃穩</td><td>前端太粗，checkpoint 還沒 ONNX 化，記憶體偏高</td></tr>
    </tbody>
  </table>

  <h2>這次實際做了什麼</h2>
  <div class="note">
    <p><strong>1. 產老師語料：</strong>用你喜歡的 IndexTTS2 reference pack，生成 16 句日常對話。輸出 manifest 在 <code>{TEACHER.relative_to(ROOT)}/manifest.json</code>。</p>
    <p><strong>2. Piper/VITS：</strong>把老師 wav 轉成 Piper LJSpeech 格式，從中文 Piper medium checkpoint warmstart fine-tune，再匯出 61MB ONNX。這次試聽使用 Piper 內建中文前端，速度是手機方向最強。</p>
    <p><strong>3. Matcha/FastSpeech-style：</strong>因上游 Matcha 字表不吃中文漢字，先把文本轉成無聲調拼音，訓練 80 step acoustic model，再接 HiFi-GAN 產波形。這條要繼續做，下一步應該改 tone pinyin/注音前端並匯出 ONNX。</p>
  </div>

  <h2>下一步</h2>
  <div class="note">
    <p><strong>先聽這份。</strong>如果 Piper 聲音方向能接受，就把 IndexTTS2 teacher corpus 擴到 500-2000 句，再做 medium 長訓和 tiny/low 架構壓縮。</p>
    <p><strong>如果 Matcha 聲音有潛力，</strong>下一步不是加 step 而已，而是修中文前端：tone pinyin 或注音符號，避免同音字和語調資訊全丟掉。</p>
  </div>
</main>
</body>
</html>
"""


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rel_copy_assets()
    (REPORT_DIR / "index.html").write_text(render(standalone=False), encoding="utf-8")
    (REPORT_DIR / "standalone.html").write_text(render(standalone=True), encoding="utf-8")
    print(REPORT_DIR / "index.html")
    print(REPORT_DIR / "standalone.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
