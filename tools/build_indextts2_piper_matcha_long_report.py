#!/usr/bin/env python3
"""Build a desktop technical report for IndexTTS2 -> Piper/Matcha long runs."""

from __future__ import annotations

import html
import json
import shutil
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
REPORT = BASE / "reports" / "indextts2_piper_matcha_long_v1"
AUDIO = REPORT / "audio"

TEACHER = BASE / "teacher_indextts2_distill_large_v1"
PIPER = BASE / "students" / "piper_indextts2_phoneme_ids_long_v7_resume_step10000"
MATCHA = BASE / "students" / "matcha_indextts2_pinyin_long_v6_step10000"
MODELS = ROOT / "models"

PIPER_ONNX = PIPER / "zh_TW-indextts2-piper-phoneme_ids-long_v7-resume-step10000-low.onnx"
PIPER_CKPT = PIPER / "checkpoints" / "epoch=39-step=10000.ckpt"
MATCHA_CKPT = MATCHA / "checkpoints" / "last.ckpt"
MATCHA_VOCODER = MODELS / "matcha" / "hifigan_T2_v1"

SAMPLES = [
    {
        "id": "01",
        "text": "你先不要急，我们慢慢来，把事情一件一件处理好。",
        "teacher": TEACHER / "audio" / "indextts2_tw_0001.wav",
        "piper": PIPER / "samples_report_v1" / "piper_long_1.wav",
        "matcha": MATCHA / "samples_report_v1" / "matcha_long_1.wav",
    },
    {
        "id": "02",
        "text": "我刚刚看了一下，应该不是你的问题，你不用太担心。",
        "teacher": TEACHER / "audio" / "indextts2_tw_0002.wav",
        "piper": PIPER / "samples_report_v1" / "piper_long_2.wav",
        "matcha": MATCHA / "samples_report_v1" / "matcha_long_2.wav",
    },
    {
        "id": "03",
        "text": "没关系啦，你先讲，我在这边听，真的不用紧张。",
        "teacher": TEACHER / "audio" / "indextts2_tw_0006.wav",
        "piper": PIPER / "samples_report_v1" / "piper_long_3.wav",
        "matcha": MATCHA / "samples_report_v1" / "matcha_long_3.wav",
    },
]


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def size_human(path: Path) -> str:
    if path.is_dir():
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        mb = total / 1024 / 1024
    else:
        mb = size_mb(path)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def rel_audio(src: Path, name: str) -> str:
    dst = AUDIO / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return html.escape(str(dst.relative_to(REPORT)))


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_for(metrics: dict, text: str) -> dict | None:
    for row in metrics.get("samples", []):
        if row.get("text") == text:
            return row
    return None


def avg(rows: list[float]) -> float:
    return sum(rows) / len(rows)


def render_audio_samples(piper_metrics: dict, matcha_metrics: dict) -> str:
    blocks: list[str] = []
    for sample in SAMPLES:
        piper_m = metric_for(piper_metrics, sample["text"])
        matcha_m = metric_for(matcha_metrics, sample["text"])
        voices = [
            (
                "IndexTTS2 teacher",
                "大模型 zero-shot clone 產生的老師語料",
                sample["teacher"],
                "teacher",
                f"音檔 {wav_seconds(sample['teacher']):.2f}s；本輪 teacher 生成時間未可靠記錄",
            ),
            (
                "Piper / VITS long",
                "10k-step 單聲音學生，ONNX 直接推論",
                sample["piper"],
                "piper",
                f"生成 {piper_m['gen_seconds']:.3f}s；音檔 {piper_m['audio_seconds']:.2f}s；RTF {piper_m['rtf']:.3f}",
            ),
            (
                "Matcha + HiFi-GAN long",
                "10k-step pinyin acoustic model + HiFi-GAN",
                sample["matcha"],
                "matcha",
                f"生成 {matcha_m['gen_seconds']:.3f}s；音檔 {matcha_m['audio_seconds']:.2f}s；RTF {matcha_m['rtf']:.3f}",
            ),
        ]
        cells = []
        for label, note, src, kind, meta in voices:
            src_rel = rel_audio(src, f"{kind}_{sample['id']}.wav")
            cells.append(
                f"""
                <div class="voice {kind}">
                  <div class="voice-title">{html.escape(label)}</div>
                  <div class="voice-note">{html.escape(note)}</div>
                  <audio controls preload="metadata" src="{src_rel}"></audio>
                  <div class="voice-meta">{html.escape(meta)}</div>
                </div>
                """
            )
        blocks.append(
            f"""
            <section class="sample">
              <div class="sample-label">Sample {sample['id']}</div>
              <h3>{html.escape(sample['text'])}</h3>
              <div class="voice-grid">{''.join(cells)}</div>
            </section>
            """
        )
    return "\n".join(blocks)


def code_block(text: str) -> str:
    return f"<pre><code>{html.escape(text.strip())}</code></pre>"


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)

    piper_metrics = load_metrics(PIPER / "samples_report_v1" / "metrics.json")
    matcha_metrics = load_metrics(MATCHA / "samples_report_v1" / "metrics.json")

    piper_gen = [s["gen_seconds"] for s in piper_metrics["samples"]]
    matcha_gen = [s["gen_seconds"] for s in matcha_metrics["samples"]]
    piper_rtf = [s["rtf"] for s in piper_metrics["samples"]]
    matcha_rtf = [s["rtf"] for s in matcha_metrics["samples"]]

    sample_html = render_audio_samples(piper_metrics, matcha_metrics)
    piper_train_cmd = f"""
cd {ROOT / 'external' / 'piper1-gpl'}
export PYTHONPATH="{ROOT / 'external' / 'piper1-gpl' / 'src'}"
{ROOT / '.venv-piper' / 'bin' / 'python'} -m piper.train fit \\
  --config {BASE / 'students' / 'piper_vits_phoneme_ids_step1000' / 'config.yaml'} \\
  --trainer.default_root_dir {PIPER} \\
  --trainer.max_steps 10000 \\
  --data.csv_path {BASE / 'datasets' / 'piper_phoneme_ids_indextts2_500_v1' / 'metadata.csv'} \\
  --data.audio_dir {BASE / 'datasets' / 'piper_phoneme_ids_indextts2_500_v1' / 'wavs'} \\
  --data.phonemes_path {BASE / 'datasets' / 'piper_phoneme_ids_indextts2_500_v1' / 'phonemes.json'} \\
  --data.cache_dir {PIPER / 'cache'} \\
  --data.config_path {PIPER / 'zh_TW-indextts2-piper-phoneme_ids-long_v7-resume-step10000-low.onnx.json'} \\
  --ckpt_path {BASE / 'students' / 'piper_indextts2_phoneme_ids_long_v6_step10000' / 'checkpoints' / 'epoch=6-step=1750.ckpt'}
"""
    piper_export_cmd = f"""
PYTHONPATH={ROOT / 'external' / 'piper1-gpl' / 'src'} \\
{ROOT / '.venv-piper' / 'bin' / 'python'} -m piper.train.export_onnx \\
  --checkpoint {PIPER_CKPT} \\
  --output-file {PIPER_ONNX}

PYTHONPATH={ROOT / 'tools'}:{ROOT / 'external' / 'piper1-gpl' / 'src'} \\
{ROOT / '.venv-piper' / 'bin' / 'python'} {ROOT / 'tools' / 'synthesize_piper_report_samples.py'} \\
  --model {PIPER_ONNX} \\
  --out-dir {PIPER / 'samples_report_v1'}
"""
    matcha_train_cmd = f"""
cd {ROOT / 'external' / 'CosyVoice' / 'third_party' / 'Matcha-TTS'}
export PYTHONPATH="{ROOT / 'external' / 'CosyVoice' / 'third_party' / 'Matcha-TTS'}"
{ROOT / '.venv-matcha' / 'bin' / 'python'} matcha/train.py \\
  data.train_filelist_path={BASE / 'datasets' / 'matcha_pinyin_indextts2_500_v1' / 'train.txt'} \\
  data.valid_filelist_path={BASE / 'datasets' / 'matcha_pinyin_indextts2_500_v1' / 'valid.txt'} \\
  data.cleaners=[] \\
  trainer.max_steps=10000 \\
  ckpt_path={MATCHA / 'checkpoints' / 'last.ckpt'}
"""
    matcha_infer_cmd = f"""
PYTHONPATH={ROOT / 'tools'}:{ROOT / 'external' / 'CosyVoice' / 'third_party' / 'Matcha-TTS'} \\
{ROOT / '.venv-matcha' / 'bin' / 'python'} {ROOT / 'tools' / 'synthesize_matcha_pinyin_long.py'} \\
  --checkpoint {MATCHA_CKPT} \\
  --vocoder {MATCHA_VOCODER} \\
  --out-dir {MATCHA / 'samples_report_v1'} \\
  --steps 10
"""

    html_doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IndexTTS2 → Piper / Matcha 長訓蒸餾報告</title>
  <style>
    :root {{
      --paper: #f6f1e8;
      --panel: #fffaf1;
      --ink: #24211d;
      --muted: #6f675c;
      --line: #ddd2c2;
      --line2: #cbbba6;
      --red: #9f2d2c;
      --tea: #51664e;
      --indigo: #485d73;
      --amber: #927247;
      --code: #2f2b26;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Noto Sans TC", "PingFang TC", sans-serif;
      font-size: 16px;
      line-height: 1.72;
      letter-spacing: 0;
    }}
    main {{
      width: min(1220px, calc(100vw - 64px));
      margin: 0 auto;
      padding: 44px 0 72px;
    }}
    header {{
      border-bottom: 1px solid var(--line2);
      padding-bottom: 28px;
      margin-bottom: 28px;
    }}
    .eyebrow {{
      color: var(--red);
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    h1 {{
      font-size: 46px;
      line-height: 1.12;
      margin: 0 0 14px;
      font-weight: 760;
      letter-spacing: 0;
    }}
    .lead {{
      max-width: 920px;
      color: var(--muted);
      font-size: 18px;
      margin: 0;
    }}
    h2 {{
      font-size: 28px;
      line-height: 1.24;
      margin: 44px 0 14px;
      letter-spacing: 0;
    }}
    h3 {{
      font-size: 19px;
      line-height: 1.45;
      margin: 4px 0 16px;
      letter-spacing: 0;
    }}
    p {{ margin: 0 0 14px; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 24px 0 10px;
    }}
    .metric, .sample, .note, .flow, .path-card {{
      background: rgba(255, 250, 241, 0.72);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{ padding: 16px; }}
    .metric b {{
      display: block;
      font-size: 25px;
      line-height: 1.15;
      margin-bottom: 4px;
    }}
    .metric span, .muted, .voice-note, .voice-meta, .small {{
      color: var(--muted);
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(255, 250, 241, 0.72);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin: 14px 0 24px;
    }}
    th, td {{
      text-align: left;
      vertical-align: top;
      padding: 12px 13px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }}
    th {{
      font-size: 12px;
      color: var(--muted);
      font-weight: 760;
      background: rgba(236, 226, 211, 0.35);
    }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }}
    pre {{
      margin: 12px 0 18px;
      padding: 14px;
      overflow-x: auto;
      border-radius: 8px;
      background: var(--code);
      color: #fbf4e8;
      line-height: 1.55;
    }}
    .sample {{
      padding: 18px;
      margin: 14px 0;
    }}
    .sample-label {{
      color: var(--red);
      font-size: 12px;
      font-weight: 760;
      margin-bottom: 2px;
    }}
    .voice-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .voice {{
      border-top: 3px solid var(--line2);
      padding-top: 10px;
      min-width: 0;
    }}
    .voice.teacher {{ border-color: var(--red); }}
    .voice.piper {{ border-color: var(--tea); }}
    .voice.matcha {{ border-color: var(--indigo); }}
    .voice-title {{
      font-weight: 760;
      margin-bottom: 2px;
    }}
    audio {{
      width: 100%;
      margin: 10px 0 4px;
    }}
    .note {{
      padding: 16px 18px;
      margin: 14px 0 24px;
    }}
    .flow {{
      padding: 18px;
      margin: 14px 0 24px;
    }}
    .flow-row {{
      display: grid;
      grid-template-columns: 1fr 42px 1fr 42px 1fr 42px 1fr;
      align-items: stretch;
      gap: 8px;
    }}
    .flow-box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fffcf6;
      min-height: 112px;
    }}
    .flow-arrow {{
      display: grid;
      place-items: center;
      color: var(--amber);
      font-size: 25px;
    }}
    .flow-title {{
      font-weight: 760;
      margin-bottom: 4px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      align-items: start;
    }}
    .path-card {{
      padding: 14px;
      margin-bottom: 12px;
    }}
    .path-card code {{
      overflow-wrap: anywhere;
    }}
    .status-good {{ color: #41633f; font-weight: 760; }}
    .status-warn {{ color: #9f5d26; font-weight: 760; }}
    .status-bad {{ color: #9f2d2c; font-weight: 760; }}
    @media (max-width: 980px) {{
      main {{ width: min(100vw - 28px, 1220px); padding-top: 28px; }}
      h1 {{ font-size: 34px; }}
      .summary-grid, .voice-grid, .two-col {{ grid-template-columns: 1fr; }}
      .flow-row {{ grid-template-columns: 1fr; }}
      .flow-arrow {{ transform: rotate(90deg); }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">2026-06-06 · local distillation report</div>
    <h1>IndexTTS2 → Piper / Matcha 長訓蒸餾報告</h1>
    <p class="lead">這份報告整理目前「用 IndexTTS2 golden voice 產生 500 句老師語料，再訓練手機候選小模型」的實驗結果。它包含真正可播放樣本、模型大小、Python CPU 推論計時、峰值記憶體、資料與訓練流程、目前失敗/限制，以及下一步該怎麼做。</p>
  </header>

  <section class="summary-grid">
    <div class="metric"><b>Piper ONNX {size_human(PIPER_ONNX)}</b><span>10k-step 長訓後匯出，可直接 ONNX 推論</span></div>
    <div class="metric"><b>{avg(piper_gen):.3f}s / 句</b><span>Piper 三句平均生成時間，不含模型初次載入</span></div>
    <div class="metric"><b>{avg(matcha_gen):.3f}s / 句</b><span>Matcha + HiFi-GAN 三句平均生成時間，不含模型初次載入</span></div>
    <div class="metric"><b>500 句</b><span>IndexTTS2 teacher corpus，單一女生聲音資料蒸餾</span></div>
  </section>

  <h2>一句話結論</h2>
  <div class="note">
    <p><strong>目前真的能往手機即時前進的是 Piper/VITS：</strong>ONNX 約 {size_human(PIPER_ONNX)}，載入約 {piper_metrics['load_seconds']:.3f}s，Python ONNX CPU 三句推論約 {avg(piper_gen):.3f}s / 句，peak RSS 約 511 MB。這個速度非常好，但它的自然度與音色上限比較低，接下來要靠你聽樣本判斷值不值得繼續大練。</p>
    <p><strong>Matcha 速度也不差，但部署包比較麻煩：</strong>acoustic checkpoint {size_human(MATCHA_CKPT)} + HiFi-GAN {size_human(MATCHA_VOCODER)}，Python CPU peak RSS 約 1.53 GB。它是非 AR / flow-matching acoustic model，比 Piper 更有自然度潛力，但目前還不是手機-ready，必須做 ONNX/CoreML/NCNN 類匯出、vocoder 量化或換更小 vocoder。</p>
  </div>

  <h2>模型對比</h2>
  <table>
    <thead>
      <tr>
        <th>模型</th>
        <th>角色</th>
        <th>部署檔大小</th>
        <th>載入時間</th>
        <th>一句生成時間</th>
        <th>Peak RSS / Python CPU</th>
        <th>狀態判斷</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>IndexTTS2 golden teacher</td>
        <td>大模型 zero-shot clone 老師；用你選的女生 reference 產 500 句</td>
        <td>本報告未重包模型；teacher corpus 音檔資料夾約 {size_human(TEACHER / 'audio')}</td>
        <td>本輪未可靠重測</td>
        <td>本輪 manifest 的 seconds 欄位不可信，故不列</td>
        <td>未重測</td>
        <td><span class="status-good">音色基準</span>，但手機離線不現實</td>
      </tr>
      <tr>
        <td>Piper / VITS long v7</td>
        <td>小型單聲音學生；中文 → phoneme ids → VITS generator</td>
        <td>ONNX {size_human(PIPER_ONNX)}；訓練 ckpt {size_human(PIPER_CKPT)}，ckpt 含 optimizer/discriminator 所以很大</td>
        <td>{piper_metrics['load_seconds']:.3f}s</td>
        <td>{avg(piper_gen):.3f}s 平均；RTF {avg(piper_rtf):.3f}</td>
        <td>535,674,880 bytes max RSS；481,182,608 bytes peak footprint</td>
        <td><span class="status-good">速度最好</span>；品質要靠聽感決定是否繼續</td>
      </tr>
      <tr>
        <td>Matcha pinyin long v6 + HiFi-GAN</td>
        <td>非 AR flow-matching acoustic student；中文 → pinyin tone chars → mel → HiFi-GAN</td>
        <td>Matcha ckpt {size_human(MATCHA_CKPT)} + vocoder {size_human(MATCHA_VOCODER)}</td>
        <td>{matcha_metrics['load_seconds']:.3f}s</td>
        <td>{avg(matcha_gen):.3f}s 平均；RTF {avg(matcha_rtf):.3f}</td>
        <td>1,639,546,880 bytes max RSS；1,545,537,456 bytes peak footprint</td>
        <td><span class="status-warn">有潛力但還不 mobile-ready</span>；需匯出/量化/換 vocoder</td>
      </tr>
    </tbody>
  </table>

  <h2>試聽對比</h2>
  <p class="muted">三欄是同一句文本：左邊老師，中間 Piper 長訓，右邊 Matcha 長訓。這裡的 teacher audio 是真正 IndexTTS2 corpus 音檔，不是 reference 原音。</p>
  {sample_html}

  <h2>整體流程圖</h2>
  <div class="flow">
    <div class="flow-row">
      <div class="flow-box">
        <div class="flow-title">1. 選聲音</div>
        <div class="small">從下載資料夾女聲片段裁 reference，排除男聲、配樂、唱歌。你最後偏好的方向是台灣女生、溫柔、清亮、低卷舌。</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-box">
        <div class="flow-title">2. 產老師語料</div>
        <div class="small">用 IndexTTS2 clone reference，生成 500 句日常對話。每筆資料有 text、teacher wav、reference metadata。</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-box">
        <div class="flow-title">3. 轉學生格式</div>
        <div class="small">Piper 轉 phoneme ids；Matcha 轉 tone-number pinyin 並計算 mel mean/std。兩邊都做 train/valid split。</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-box">
        <div class="flow-title">4. 長訓學生</div>
        <div class="small">Piper 從 v6 1,750 step resume 到 10,000 step；Matcha 從 last checkpoint resume 到 max_steps=10,000。</div>
      </div>
    </div>
  </div>

  <h2>這算蒸餾嗎？</h2>
  <div class="note">
    <p><strong>算 data distillation / teacher-student voice distillation。</strong>老師模型 IndexTTS2 先產生高品質 synthetic labels，也就是 500 句「文字 → 老師聲音 wav」。學生模型 Piper / Matcha 不直接看原始 reference，也不呼叫 IndexTTS2，而是學這批老師 wav 的音色、口音、語尾與節奏。</p>
    <p><strong>但它還不是 logits distillation。</strong>我們目前沒有拿 teacher 的 hidden states、token probabilities、duration posterior 或 mel distribution 做 KL / feature matching。TTS 開源模型之間架構差太多，這類蒸餾通常要改訓練 loop。現階段是實務上最直接的「資料蒸餾 + 單聲音 fine-tune」。</p>
  </div>

  <h2>資料與前處理</h2>
  <table>
    <thead>
      <tr><th>項目</th><th>內容</th><th>路徑</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Teacher corpus</td>
        <td>IndexTTS2 產生 500 句日常對話。音檔 22.05kHz wav。</td>
        <td><code>{TEACHER}</code></td>
      </tr>
      <tr>
        <td>Piper dataset</td>
        <td>中文文本先轉成 Piper 可吃的 phoneme / phoneme_ids，訓練 VITS generator/discriminator。</td>
        <td><code>{BASE / 'datasets' / 'piper_phoneme_ids_indextts2_500_v1'}</code></td>
      </tr>
      <tr>
        <td>Matcha dataset</td>
        <td>中文文本轉 pypinyin Style.TONE3，例如 <code>ni3 xian1 bu4 yao4...</code>；Matcha config 使用 <code>cleaners: []</code>，推論也必須用同一條前端。</td>
        <td><code>{BASE / 'datasets' / 'matcha_pinyin_indextts2_500_v1'}</code></td>
      </tr>
    </tbody>
  </table>

  <h2>模型細節</h2>
  <div class="two-col">
    <div class="path-card">
      <h3>Piper / VITS</h3>
      <p>Piper 的核心是 VITS 類模型：text/phoneme encoder、duration/alignment、flow、decoder/generator，訓練時搭配 discriminator 做 adversarial loss。它的優點是部署簡單，ONNX generator 可以直接在手機跑；缺點是單聲音 fine-tune 要把音色、韻律、情緒都壓進相對小的模型，品質上限通常不如大 clone 模型。</p>
      <p class="small">本輪訓練 checkpoint 很大，主要因為 Lightning checkpoint 包了 generator、discriminator、optimizer states；真正部署只需要 ONNX。</p>
    </div>
    <div class="path-card">
      <h3>Matcha + HiFi-GAN</h3>
      <p>Matcha 是 conditional flow matching acoustic model：文字條件進 encoder，decoder 透過 ODE steps 生成 mel，再由 HiFi-GAN vocoder 轉 waveform。它是非 AR，理論上速度比大 AR/LLM TTS 好；但手機部署不是一個檔案，要處理 acoustic model + vocoder 的匯出、量化與記憶體。</p>
      <p class="small">這輪為了避開 espeak，使用 pinyin char frontend。這讓工程可控，但語音自然度可能被「字母級拼音」限制。</p>
    </div>
  </div>

  <h2>實際訓練與推論命令</h2>
  <h3>Piper 長訓 resume</h3>
  {code_block(piper_train_cmd)}
  <h3>Piper ONNX 匯出與推論</h3>
  {code_block(piper_export_cmd)}
  <h3>Matcha 長訓 resume</h3>
  {code_block(matcha_train_cmd)}
  <h3>Matcha pinyin 推論</h3>
  {code_block(matcha_infer_cmd)}

  <h2>訓練結果紀錄</h2>
  <table>
    <thead>
      <tr><th>模型</th><th>結果</th><th>關鍵 log</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Piper long v7</td>
        <td><span class="status-good">完成 10,000 steps</span>，log 顯示 <code>Trainer.fit stopped: max_steps=10000 reached</code>，最後 checkpoint <code>epoch=39-step=10000.ckpt</code>。</td>
        <td><code>{BASE / 'logs' / 'piper_indextts2_phoneme_ids_long_v7_resume_step10000_screen.log'}</code></td>
      </tr>
      <tr>
        <td>Matcha long v6</td>
        <td><span class="status-good">完成 max_steps=10,000</span>，log final step 9999，loss/train_epoch 1.619。注意：<code>last.ckpt</code> metadata 顯示 global_step 9441，因 checkpoint cadence 沒剛好存到 final step；訓練 log 才是完成依據。</td>
        <td><code>{BASE / 'logs' / 'matcha_indextts2_pinyin_long_v6_step10000_resume_screen.log'}</code></td>
      </tr>
    </tbody>
  </table>

  <h2>目前限制與下一步</h2>
  <table>
    <thead>
      <tr><th>問題</th><th>原因</th><th>下一步</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Piper 速度很好，但可能不像 / 不自然</td>
        <td>VITS 小模型壓縮太重；500 句 synthetic teacher 也可能不足以穩住韻律。</td>
        <td>先人工聽本報告三句。若文字正確但音色不足，做 2k–5k 句 teacher corpus + speaker embedding/feature loss；若文字錯或雜訊，先檢查 phoneme frontend 與資料品質。</td>
      </tr>
      <tr>
        <td>Matcha peak RSS 太高</td>
        <td>Python + PyTorch + acoustic + HiFi-GAN 都在記憶體裡；手機不能照這條 runtime。</td>
        <td>匯出 acoustic ONNX / CoreML，vocoder 換小 HiFi-GAN、MelGAN、BigVGAN tiny 或 EnCodec 類；再量化。</td>
      </tr>
      <tr>
        <td>這還不是最嚴格的 logits distillation</td>
        <td>IndexTTS2、Piper、Matcha 架構不同，直接蒸 hidden/logits 要改各模型 training loop。</td>
        <td>若你要真正研究型蒸餾，下一階段做 mel-level loss + speaker embedding loss + ASR text consistency loss，比單純訓 wav 更穩。</td>
      </tr>
      <tr>
        <td>Teacher generation time 不列</td>
        <td>目前 teacher manifest 的 seconds 欄位不可信；它看起來像錯誤紀錄，不是生成時間也不是音檔長度。</td>
        <td>如果要完整 benchmark teacher，需要重新跑 IndexTTS2 三句並用同一支 profiler 記錄 load/gen/RSS。</td>
      </tr>
    </tbody>
  </table>

  <h2>重要檔案路徑</h2>
  <div class="two-col">
    <div>
      <div class="path-card"><strong>本報告</strong><br><code>{REPORT / 'index.html'}</code></div>
      <div class="path-card"><strong>Teacher corpus</strong><br><code>{TEACHER}</code></div>
      <div class="path-card"><strong>Piper ONNX</strong><br><code>{PIPER_ONNX}</code></div>
      <div class="path-card"><strong>Piper samples</strong><br><code>{PIPER / 'samples_report_v1'}</code></div>
    </div>
    <div>
      <div class="path-card"><strong>Matcha checkpoint</strong><br><code>{MATCHA_CKPT}</code></div>
      <div class="path-card"><strong>Matcha samples</strong><br><code>{MATCHA / 'samples_report_v1'}</code></div>
      <div class="path-card"><strong>新增 Piper 推論腳本</strong><br><code>{ROOT / 'tools' / 'synthesize_piper_report_samples.py'}</code></div>
      <div class="path-card"><strong>新增 Matcha 推論腳本</strong><br><code>{ROOT / 'tools' / 'synthesize_matcha_pinyin_long.py'}</code></div>
    </div>
  </div>

  <h2>附錄：這次修了什麼</h2>
  <div class="note">
    <p>1. 清掉因 runaway <code>git add</code> 造成的巨大 temporary pack，磁碟已恢復可用；並加入 <code>.gitignore</code> 排除模型、venv、音訊、distillation artifacts。</p>
    <p>2. Piper 長訓 checkpoint 已匯出 ONNX，並用本機 pypinyin/Piper frontend 生成三句樣本。</p>
    <p>3. Matcha CLI 原本硬走 espeak 英文 cleaner，與本訓練資料不一致；已新增專用 pinyin inference 腳本，確保訓練/推論前端一致。</p>
    <p>4. Matcha source 的 checkpoint loading 因 PyTorch 2.6 <code>weights_only=True</code> 預設而失敗；已在本地 source 以可信 checkpoint 模式改成 <code>weights_only=False</code>。</p>
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
