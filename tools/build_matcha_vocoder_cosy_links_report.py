#!/usr/bin/env python3
"""Build a compact report with Cosy links and Matcha vocoder auditions."""

from __future__ import annotations

import html
import json
import shutil
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
REPORT = BASE / "reports" / "matcha_vocoder_cosy_links_v1"
AUDIO = REPORT / "audio"
MATCHA = BASE / "students" / "matcha_indextts2_pinyin_long_v6_step10000"
TEACHER = BASE / "teacher_indextts2_distill_large_v1"

COSY_LINKS = [
    ("Cosy teacher 500 selection", BASE / "reports" / "cosy_teacher_500_selection_v1" / "cosy_teacher_500_selection_v1.html", "之前用來選 500 句老師語料 / golden teacher 的頁面。"),
    ("Cosy teacher 500 cloud audio", BASE / "reports" / "cosy_teacher_500_cloud_audio_v1" / "index.html", "500 句老師語料總覽，本機 HTML。"),
    ("Cosy teacher cloud index", BASE / "reports" / "cosy_teacher_500_cloud_audio_v1" / "cloud_index.html", "雲端/分頁版本索引。"),
    ("Cosy vs ZipVoice 4-step", BASE / "reports" / "cosy_vs_zipvoice_true_distill_4step_v1" / "cosy_vs_zipvoice_true_distill_4step_v1_standalone.html", "你之前覺得 4-step 表現不夠穩的對比報告。"),
    ("Golden Cosy ZipVoice progress", BASE / "reports" / "golden_cosy_zipvoice_distillation_progress_v1" / "golden_cosy_zipvoice_distillation_progress_v1_standalone.html", "golden teacher 到 ZipVoice 蒸餾進度。"),
    ("Cosy → ZipVoice method", BASE / "reports" / "cosy_zipvoice_true_distill_method_v1" / "cosy_zipvoice_true_distill_method_v1_standalone.html", "蒸餾方法說明。"),
    ("Qwen / Cosy / ZipVoice pipeline", BASE / "reports" / "qwen_cosy_zipvoice_pipeline_v1" / "qwen_cosy_zipvoice_pipeline_v1.html", "從 Qwen 到 Cosy/ZipVoice 的路徑圖。"),
]

SAMPLES = [
    ("01", "你先不要急，我们慢慢来，把事情一件一件处理好。", "indextts2_tw_0001.wav", "matcha_long_1.wav", "bigvgan_1.wav"),
    ("02", "我刚刚看了一下，应该不是你的问题，你不用太担心。", "indextts2_tw_0002.wav", "matcha_long_2.wav", "bigvgan_2.wav"),
    ("03", "没关系啦，你先讲，我在这边听，真的不用紧张。", "indextts2_tw_0006.wav", "matcha_long_3.wav", "bigvgan_3.wav"),
]

VARIANTS = [
    ("baseline", "HiFi-GAN T2 baseline", MATCHA / "samples_report_v1", "上一份報告 baseline，10-step temp 0.667。"),
    ("spacing", "HiFi-GAN T2 spacing fix", MATCHA / "tuning_spacing_s16_t050_r090", "16-step temp 0.50 rate 0.90，針對字距太開。"),
    ("univ", "HiFi-GAN Univ", MATCHA / "tuning_vocoder_univ_s16_t050_r090", "同一個 Matcha mel，換官方 universal HiFi-GAN。"),
    ("bigvgan", "BigVGAN v2 22k fmax8k", MATCHA / "tuning_vocoder_bigvgan_s16_t050_r090", "大 vocoder 替換試聽，CPU 很慢但可測清晰度。"),
]


def rel(src: Path, name: str) -> str:
    AUDIO.mkdir(parents=True, exist_ok=True)
    dst = AUDIO / name
    shutil.copy2(src, dst)
    return html.escape(str(dst.relative_to(REPORT)))


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def metrics_for(directory: Path) -> dict | None:
    path = directory / "metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)

    link_rows = []
    for title, path, note in COSY_LINKS:
        if path.exists():
            link_rows.append(
                f"<tr><td>{html.escape(title)}</td><td><a href=\"{html.escape(str(path))}\">{html.escape(str(path))}</a></td><td>{html.escape(note)}</td></tr>"
            )

    speed_rows = []
    for vid, label, directory, note in VARIANTS:
        m = metrics_for(directory)
        if not m:
            speed = "未記錄"
            load = "未記錄"
        else:
            speed = f"{sum(s['gen_seconds'] for s in m['samples']) / len(m['samples']):.3f}s"
            load = f"{m.get('load_seconds', 0):.3f}s"
        speed_rows.append(f"<tr><td>{label}</td><td>{load}</td><td>{speed}</td><td>{html.escape(note)}</td></tr>")

    sample_blocks = []
    for sample_no, text, teacher_name, matcha_name, bigvgan_name in SAMPLES:
        idx = int(sample_no)
        teacher_src = rel(TEACHER / "audio" / teacher_name, f"teacher_{sample_no}.wav")
        cells = [
            f"""
            <div class="voice teacher">
              <div class="voice-title">IndexTTS2 teacher</div>
              <div class="voice-note">音色目標</div>
              <audio controls preload="metadata" src="{teacher_src}"></audio>
              <div class="voice-meta">{wav_seconds(TEACHER / 'audio' / teacher_name):.2f}s</div>
            </div>
            """
        ]
        for vid, label, directory, note in VARIANTS:
            filename = bigvgan_name if vid == "bigvgan" else matcha_name
            src = rel(directory / filename, f"{vid}_{sample_no}.wav")
            m = metrics_for(directory)
            meta = note
            if m:
                s = m["samples"][idx - 1]
                meta = f"{note} · gen {s['gen_seconds']:.3f}s / wav {s['audio_seconds']:.2f}s"
            cells.append(
                f"""
                <div class="voice">
                  <div class="voice-title">{html.escape(label)}</div>
                  <div class="voice-note">{html.escape(note)}</div>
                  <audio controls preload="metadata" src="{src}"></audio>
                  <div class="voice-meta">{html.escape(meta)}</div>
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

    doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cosy 報告連結 + Matcha vocoder 試聽</title>
  <style>
    :root {{ --paper:#f6f1e8; --panel:#fffaf1; --ink:#24211d; --muted:#6f675c; --line:#ddd2c2; --red:#9f2d2c; --blue:#485d73; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans TC","PingFang TC",sans-serif; font-size:16px; line-height:1.72; letter-spacing:0; }}
    main {{ width:min(1240px, calc(100vw - 64px)); margin:0 auto; padding:44px 0 72px; }}
    header {{ border-bottom:1px solid #cbbba6; padding-bottom:26px; margin-bottom:28px; }}
    .eyebrow {{ color:var(--red); font-size:13px; font-weight:760; margin-bottom:10px; }}
    h1 {{ font-size:43px; line-height:1.12; margin:0 0 12px; letter-spacing:0; }}
    h2 {{ font-size:27px; margin:42px 0 14px; letter-spacing:0; }}
    h3 {{ font-size:19px; margin:4px 0 15px; letter-spacing:0; }}
    .lead,.muted,.voice-note,.voice-meta {{ color:var(--muted); }}
    .lead {{ font-size:18px; max-width:940px; }}
    table {{ width:100%; border-collapse:collapse; background:rgba(255,250,241,.78); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th,td {{ padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; font-size:14px; }}
    th {{ color:var(--muted); font-size:12px; background:rgba(236,226,211,.35); }}
    tr:last-child td {{ border-bottom:0; }}
    a {{ color:#7a2a2a; text-decoration:none; overflow-wrap:anywhere; }}
    .note,.sample {{ background:rgba(255,250,241,.78); border:1px solid var(--line); border-radius:8px; }}
    .note {{ padding:16px 18px; margin:14px 0 24px; }}
    .sample {{ padding:18px; margin:14px 0; }}
    .sample-label {{ color:var(--red); font-weight:760; font-size:12px; }}
    .voice-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
    .voice {{ border-top:3px solid var(--blue); padding-top:10px; min-width:0; }}
    .voice.teacher {{ border-color:var(--red); }}
    .voice-title {{ font-weight:760; }}
    .voice-note,.voice-meta {{ font-size:13px; }}
    audio {{ width:100%; margin:9px 0 4px; }}
    @media (max-width:980px) {{ main {{ width:min(100vw - 28px,1240px); padding-top:28px; }} h1 {{ font-size:34px; }} .voice-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">2026-06-06 · Cosy links / Matcha vocoder audition</div>
    <h1>Cosy 報告連結 + Matcha vocoder 試聽</h1>
    <p class="lead">這頁先把之前 Cosy 相關報告集中起來，下面放 Matcha 同一批 mel 換不同 vocoder / spacing 修正的試聽。BigVGAN 是大模型替換測試，不是手機候選。</p>
  </header>
  <h2>Cosy 之前報告</h2>
  <table><thead><tr><th>報告</th><th>連結</th><th>用途</th></tr></thead><tbody>{''.join(link_rows)}</tbody></table>
  <h2>Vocoder / spacing 速度</h2>
  <table><thead><tr><th>版本</th><th>載入</th><th>平均生成</th><th>說明</th></tr></thead><tbody>{''.join(speed_rows)}</tbody></table>
  <div class="note">
    <p><strong>目前最值得聽：</strong>HiFi-GAN T2 spacing fix、HiFi-GAN Univ、BigVGAN。BigVGAN CPU 很慢，若它明顯比較清楚，代表 vocoder 是大瓶頸；若沒差，就要回去改 Matcha acoustic / dataset。</p>
  </div>
  <h2>試聽</h2>
  {''.join(sample_blocks)}
</main>
</body>
</html>
"""
    (REPORT / "index.html").write_text(doc, encoding="utf-8")
    print(REPORT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
