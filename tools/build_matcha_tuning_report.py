#!/usr/bin/env python3
"""Build an audition report for Matcha inference tuning variants."""

from __future__ import annotations

import html
import json
import shutil
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
MATCHA = BASE / "students" / "matcha_indextts2_pinyin_long_v6_step10000"
TEACHER = BASE / "teacher_indextts2_distill_large_v1"
REPORT = BASE / "reports" / "matcha_tuning_v1"
AUDIO = REPORT / "audio"

SAMPLES = [
    ("01", "你先不要急，我们慢慢来，把事情一件一件处理好。", "indextts2_tw_0001.wav", "matcha_long_1.wav"),
    ("02", "我刚刚看了一下，应该不是你的问题，你不用太担心。", "indextts2_tw_0002.wav", "matcha_long_2.wav"),
    ("03", "没关系啦，你先讲，我在这边听，真的不用紧张。", "indextts2_tw_0006.wav", "matcha_long_3.wav"),
]

VARIANTS = [
    {
        "id": "current",
        "label": "Current baseline",
        "dir": MATCHA / "samples_report_v1",
        "steps": 10,
        "temperature": 0.667,
        "rate": 1.00,
        "note": "上一份報告使用的 baseline。",
    },
    {
        "id": "s10_t050_r100",
        "label": "10-step, temp 0.50",
        "dir": MATCHA / "tuning_s10_t050_r100",
        "steps": 10,
        "temperature": 0.50,
        "rate": 1.00,
        "note": "保留速度，降低隨機性，通常會比較穩。",
    },
    {
        "id": "s16_t050_r100",
        "label": "16-step, temp 0.50",
        "dir": MATCHA / "tuning_s16_t050_r100",
        "steps": 16,
        "temperature": 0.50,
        "rate": 1.00,
        "note": "第一優先候選：穩定度與速度折衷。",
    },
    {
        "id": "s24_t050_r100",
        "label": "24-step, temp 0.50",
        "dir": MATCHA / "tuning_s24_t050_r100",
        "steps": 24,
        "temperature": 0.50,
        "rate": 1.00,
        "note": "品質上限測試，速度較慢。",
    },
    {
        "id": "s16_t040_r100",
        "label": "16-step, temp 0.40",
        "dir": MATCHA / "tuning_s16_t040_r100",
        "steps": 16,
        "temperature": 0.40,
        "rate": 1.00,
        "note": "更保守，可能更穩也可能變平。",
    },
    {
        "id": "s16_t055_r095",
        "label": "16-step, temp 0.55, faster",
        "dir": MATCHA / "tuning_s16_t055_r095",
        "steps": 16,
        "temperature": 0.55,
        "rate": 0.95,
        "note": "稍快一點，測試日常感。",
    },
    {
        "id": "s16_t055_r105",
        "label": "16-step, temp 0.55, slower",
        "dir": MATCHA / "tuning_s16_t055_r105",
        "steps": 16,
        "temperature": 0.55,
        "rate": 1.05,
        "note": "第二優先候選：稍慢，可能更溫柔自然。",
    },
    {
        "id": "s24_t055_r105",
        "label": "24-step, temp 0.55, slower",
        "dir": MATCHA / "tuning_s24_t055_r105",
        "steps": 24,
        "temperature": 0.55,
        "rate": 1.05,
        "note": "高成本候選，用來聽 24-step 是否值得。",
    },
]


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def rel_audio(src: Path, name: str) -> str:
    AUDIO.mkdir(parents=True, exist_ok=True)
    dst = AUDIO / name
    shutil.copy2(src, dst)
    return html.escape(str(dst.relative_to(REPORT)))


def load_metrics(variant: dict) -> dict:
    path = variant["dir"] / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def variant_wav(variant: dict, sample_idx: int) -> Path:
    if variant["id"] == "current":
        return variant["dir"] / f"matcha_long_{sample_idx}.wav"
    return variant["dir"] / f"matcha_long_{sample_idx}.wav"


def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)
    metrics = {v["id"]: load_metrics(v) for v in VARIANTS}

    rows = []
    for variant in VARIANTS:
        m = metrics[variant["id"]]
        gen = [s["gen_seconds"] for s in m["samples"]]
        rtf = [s["rtf"] for s in m["samples"]]
        rows.append(
            f"""
            <tr>
              <td>{html.escape(variant['label'])}</td>
              <td>{variant['steps']}</td>
              <td>{variant['temperature']:.3g}</td>
              <td>{variant['rate']:.2f}</td>
              <td>{avg(gen):.3f}s</td>
              <td>{avg(rtf):.3f}</td>
              <td>{html.escape(variant['note'])}</td>
            </tr>
            """
        )

    sample_blocks = []
    for sample_no, text, teacher_name, matcha_name in SAMPLES:
        idx = int(sample_no)
        teacher_src = rel_audio(TEACHER / "audio" / teacher_name, f"teacher_{sample_no}.wav")
        cells = [
            f"""
            <div class="voice teacher">
              <div class="voice-title">IndexTTS2 teacher</div>
              <div class="voice-note">音色目標</div>
              <audio controls preload="metadata" src="{teacher_src}"></audio>
              <div class="voice-meta">音檔 {wav_seconds(TEACHER / 'audio' / teacher_name):.2f}s</div>
            </div>
            """
        ]
        for variant in VARIANTS:
            src_path = variant_wav(variant, idx)
            src = rel_audio(src_path, f"{variant['id']}_{sample_no}.wav")
            sample_metric = metrics[variant["id"]]["samples"][idx - 1]
            cells.append(
                f"""
                <div class="voice">
                  <div class="voice-title">{html.escape(variant['label'])}</div>
                  <div class="voice-note">steps {variant['steps']} · temp {variant['temperature']:.3g} · rate {variant['rate']:.2f}</div>
                  <audio controls preload="metadata" src="{src}"></audio>
                  <div class="voice-meta">生成 {sample_metric['gen_seconds']:.3f}s · 音檔 {sample_metric['audio_seconds']:.2f}s</div>
                </div>
                """
            )
        sample_blocks.append(
            f"""
            <section class="sample">
              <div class="sample-label">Sample {sample_no}</div>
              <h3>{html.escape(text)}</h3>
              <div class="voice-grid">{''.join(cells)}</div>
            </section>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Matcha TTS 調參試聽 v1</title>
  <style>
    :root {{
      --paper:#f6f1e8; --panel:#fffaf1; --ink:#24211d; --muted:#6f675c;
      --line:#ddd2c2; --line2:#cbbba6; --red:#9f2d2c; --tea:#51664e; --blue:#485d73;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; background:var(--paper); color:var(--ink);
      font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans TC","PingFang TC",sans-serif;
      font-size:16px; line-height:1.72; letter-spacing:0;
    }}
    main {{ width:min(1240px, calc(100vw - 64px)); margin:0 auto; padding:44px 0 72px; }}
    header {{ border-bottom:1px solid var(--line2); padding-bottom:26px; margin-bottom:28px; }}
    .eyebrow {{ color:var(--red); font-size:13px; font-weight:760; margin-bottom:10px; }}
    h1 {{ font-size:44px; line-height:1.12; margin:0 0 12px; letter-spacing:0; }}
    h2 {{ font-size:27px; margin:42px 0 14px; letter-spacing:0; }}
    h3 {{ font-size:19px; margin:4px 0 15px; letter-spacing:0; }}
    .lead {{ color:var(--muted); font-size:18px; max-width:940px; margin:0; }}
    .note,.sample {{ background:rgba(255,250,241,.78); border:1px solid var(--line); border-radius:8px; }}
    .note {{ padding:16px 18px; margin:14px 0 24px; }}
    table {{ width:100%; border-collapse:collapse; background:rgba(255,250,241,.78); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th,td {{ padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; font-size:14px; }}
    th {{ color:var(--muted); font-size:12px; background:rgba(236,226,211,.35); }}
    tr:last-child td {{ border-bottom:0; }}
    .sample {{ padding:18px; margin:14px 0; }}
    .sample-label {{ color:var(--red); font-weight:760; font-size:12px; }}
    .voice-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
    .voice {{ border-top:3px solid var(--blue); padding-top:10px; min-width:0; }}
    .voice.teacher {{ border-color:var(--red); }}
    .voice-title {{ font-weight:760; }}
    .voice-note,.voice-meta,.muted {{ color:var(--muted); font-size:13px; }}
    audio {{ width:100%; margin:9px 0 4px; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px; }}
    @media (max-width:980px) {{
      main {{ width:min(100vw - 28px,1240px); padding-top:28px; }}
      h1 {{ font-size:34px; }}
      .voice-grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">2026-06-06 · Matcha inference tuning</div>
    <h1>Matcha TTS 調參試聽 v1</h1>
    <p class="lead">Piper 先放掉；這頁只比較 Matcha 長訓 checkpoint 在不同推論參數下的聲音。這一輪沒有重訓，只改 ODE steps、temperature、speaking rate，用來挑下一個可用 baseline。</p>
  </header>
  <h2>我建議先聽哪幾個</h2>
  <div class="note">
    <p><strong>優先 A：16-step, temp 0.50。</strong>比 baseline 多一點計算，但降低隨機性，理論上比較穩。</p>
    <p><strong>優先 B：16-step, temp 0.55, slower。</strong>稍微放慢，可能更接近溫柔台灣女生的語尾，不會像趕著念完。</p>
    <p><strong>24-step 只當品質上限測試。</strong>它平均接近 0.9s / 句，比 16-step 更慢；除非你聽起來明顯好很多，否則不值得放進手機路線。</p>
  </div>
  <h2>參數與速度</h2>
  <table>
    <thead><tr><th>版本</th><th>steps</th><th>temperature</th><th>語速 scale</th><th>平均生成</th><th>RTF</th><th>用途</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>試聽</h2>
  <p class="muted">每個 sample 都先放 teacher，再放 baseline 和全部候選。請直接挑「最不像機器、最像溫柔女生」的版本。</p>
  {''.join(sample_blocks)}
  <h2>下一步優化方向</h2>
  <div class="note">
    <p><strong>短期：</strong>選定本頁最佳推論參數，固定成新的 Matcha baseline，然後重生更多日常句子。</p>
    <p><strong>中期：</strong>繼續長訓 Matcha 到 30k–50k steps，並把 train/dev 中明顯不好的 teacher wav 移掉。這通常比亂調推論參數更有效。</p>
    <p><strong>大幅改善音質：</strong>fine-tune 或替換 vocoder。目前 HiFi-GAN 是通用 vocoder，不是針對這個女生聲音/teacher corpus 訓練。若要更清晰、少假聲，vocoder 是下一個大槓桿。</p>
  </div>
</main>
</body>
</html>
"""
    (REPORT / "index.html").write_text(html_doc, encoding="utf-8")
    print(REPORT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
