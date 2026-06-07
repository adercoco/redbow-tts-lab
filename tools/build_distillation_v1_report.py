#!/usr/bin/env python3
"""Build the current distillation experiment status report."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
CORPUS = BASE / "teacher_qwen3_1p7b_distill_v1"
OUT = BASE / "reports" / "distillation_v1"
ZIPVOICE_REPORT = BASE / "reports" / "zipvoice_three_way"
LINE_ID = "distill_0002"
LINE_TEXT = "这样听起来有比较像台湾人吗？我希望尾音自然一点，不要太卷舌。"


def rel(path: Path, base: Path = OUT) -> str:
    return path.relative_to(base).as_posix()


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    seen: set[tuple[int, int]] = set()
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            stat = item.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen:
                continue
            seen.add(inode)
            total += stat.st_blocks * 512 if stat.st_blocks else stat.st_size
    return total


def fmt_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{amount:.1f} GB"


def audio_duration(path: Path) -> float:
    return float(sf.info(path).duration)


def corpus_stats() -> dict:
    rows = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    ok = [row for row in rows if row.get("status") == "ok" and row.get("audio")]
    total = sum(audio_duration(Path(row["audio"])) for row in ok)
    teacher_line = next(row for row in ok if row["id"] == LINE_ID)
    return {
        "rows": len(rows),
        "ok": len(ok),
        "total_seconds": total,
        "total_hours": total / 3600,
        "avg_audio_seconds": total / len(ok),
        "avg_generation_seconds": sum(float(row.get("seconds", 0.0)) for row in ok) / len(ok),
        "line_generation_seconds": float(teacher_line.get("seconds", 0.0)),
    }


def copy_teacher_audio() -> Path:
    src = CORPUS / "audio" / f"{LINE_ID}.wav"
    dst = OUT / "audio" / f"teacher_qwen3_1p7b_{LINE_ID}.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def copy_report_audio(src: Path) -> Path:
    dst = OUT / "audio" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    return dst


def build_html(audio_prefix: str = "audio") -> str:
    stats = corpus_stats()
    teacher_audio = copy_teacher_audio()
    zipvoice_audio = OUT / "audio" / "zipvoice_zh_min_teacher_ref_0002.wav"
    f5_audio = OUT / "audio" / "f5_smoke4_update2_0002.wav"
    f5_update20_audio = OUT / "audio" / "f5_distill500_update20_0002.wav"
    f5_update60_audio = OUT / "audio" / "f5_distill500_update60_0002.wav"
    full_dataset = ROOT / ".venv-f5" / "lib" / "python3.12" / "data" / "taiwan_low_r_distill_v1_pinyin"
    smoke_ckpt = ROOT / ".venv-f5" / "lib" / "python3.12" / "ckpts" / "taiwan_low_r_distill_smoke4"
    smoke_last = smoke_ckpt / "model_last.pt"
    update20_ckpt = ROOT / ".venv-f5" / "lib" / "python3.12" / "ckpts" / "taiwan_low_r_distill_v1"
    update20_last = update20_ckpt / "model_last.pt"
    zipvoice_pack = ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min"
    vocoder = ROOT / "models" / "sherpa" / "vocos_24khz.onnx"
    piper_tiny = BASE / "students" / "piper_tiny_distill_v1_300"
    piper_tiny_onnx = piper_tiny / "taiwan_low_r_piper_tiny_distill_v1_300.onnx"
    piper_tiny_ckpt = piper_tiny / "checkpoints" / "epoch=2-step=300.ckpt"
    piper_tiny_audio = copy_report_audio(piper_tiny / "piper_tiny_300_0002.wav")
    piper_medium = BASE / "students" / "piper_medium_warmstart_100"
    piper_medium_onnx = piper_medium / "taiwan_low_r_piper_medium_warmstart_100.onnx"
    piper_medium_ckpt = piper_medium / "checkpoints" / "epoch=3270-step=246.ckpt"
    piper_medium_audio = copy_report_audio(piper_medium / "piper_medium_warm_0002.wav")
    for sample_name in [
        "piper_tiny_300_new_a.wav",
        "piper_tiny_300_new_b.wav",
        "piper_medium_warm_new_a.wav",
        "piper_medium_warm_new_b.wav",
    ]:
        source_dir = piper_tiny if sample_name.startswith("piper_tiny") else piper_medium
        copy_report_audio(source_dir / sample_name)

    rows = [
        {
            "name": "Qwen3 1.7B teacher",
            "status": "老師原聲",
            "audio": f"{audio_prefix}/{teacher_audio.name}",
            "model": "Qwen3-TTS 1.7B VoiceDesign 4bit",
            "size": "約 2.2GB runtime / cache",
            "memory": "2.20GB peak",
            "time": f"{stats['line_generation_seconds']:.2f}s",
            "mobile": "不適合手機端即時跑",
            "note": "這是要被蒸餾的老師聲音，聲線目標以它為準。",
        },
        {
            "name": "ZipVoice int8 zh-min",
            "status": "手機優先 baseline",
            "audio": f"{audio_prefix}/{zipvoice_audio.name}",
            "model": "Sherpa-ONNX ZipVoice int8 + Vocos",
            "size": fmt_bytes(size_bytes(zipvoice_pack) + size_bytes(vocoder)),
            "memory": "806MB peak / single sample",
            "time": "1.31s generation / 1.98s process",
            "mobile": "目前最像手機可落地路線",
            "note": "這還是 zero-shot voice prompt，不是完整蒸餾；優點是小、快、ONNX 路線清楚。",
        },
        {
            "name": "Piper tiny 500 / 300 steps",
            "status": "真小模型 from scratch",
            "audio": f"{audio_prefix}/{piper_tiny_audio.name}",
            "model": "Piper/VITS tiny, trained from scratch on 500 teacher wavs",
            "size": f"{fmt_bytes(size_bytes(piper_tiny_onnx))} ONNX / {fmt_bytes(size_bytes(piper_tiny_ckpt))} training ckpt",
            "memory": "128.8MB peak RSS / piper subprocess",
            "time": "0.44s process / same line",
            "mobile": "很小、很快；但 from-scratch 300 steps 還不像",
            "note": "部署大小已證明可做到 14MB 級，但 proxy score 只有 0.694；需要 warmstart 或更長訓練才有音色意義。",
        },
        {
            "name": "Piper medium warmstart",
            "status": "目前最佳完整蒸餾小模型",
            "audio": f"{audio_prefix}/{piper_medium_audio.name}",
            "model": "Piper/VITS medium, zh_CN huayan checkpoint fine-tuned on teacher wavs",
            "size": f"{fmt_bytes(size_bytes(piper_medium_onnx))} ONNX / {fmt_bytes(size_bytes(piper_medium_ckpt))} training ckpt",
            "memory": "266.0MB peak RSS / piper subprocess",
            "time": "0.63s process / same line",
            "mobile": "可手機部署；比 tiny 大，但仍遠小於 Qwen/F5",
            "note": "從官方中文 Piper checkpoint warmstart 後 proxy score 0.923，高於 F5 update-20 的 0.872；這是目前最合理的小模型蒸餾分支。",
        },
        {
            "name": "F5 smoke checkpoint",
            "status": "訓練管線已通",
            "audio": f"{audio_prefix}/{f5_audio.name}",
            "model": "F5TTS_v1_Base fine-tune smoke",
            "size": f"{fmt_bytes(size_bytes(smoke_last))} checkpoint / {fmt_bytes(size_bytes(smoke_ckpt))} folder",
            "memory": "7.43GB inference peak / 9.37GB train peak",
            "time": "9.28s generation / 15.03s process",
            "mobile": "不適合手機，只作蒸餾流程證明",
            "note": "只用 4 句、2 updates，不能代表最終音色品質；它證明 teacher data 可以進 student 訓練並輸出 wav。",
        },
        {
            "name": "F5 distilled 500 / update 20",
            "status": "實際蒸餾樣本",
            "audio": f"{audio_prefix}/{f5_update20_audio.name}",
            "model": "F5TTS_v1_Base fine-tuned on 500 teacher wavs",
            "size": f"{fmt_bytes(size_bytes(update20_last))} checkpoint / {fmt_bytes(size_bytes(update20_ckpt))} folder",
            "memory": "GB-class; CPU train succeeded, not mobile-suitable",
            "time": "15-17s generation / sample on CPU",
            "mobile": "不是手機終點，但是真正吃過 500 條 teacher corpus 的 student",
            "note": "這是目前第一個有效 distilled student 結果。訓練只有 20 updates，聲線可聽但不應視為最終品質。",
        },
        {
            "name": "F5 distilled 500 / update 60",
            "status": "續訓樣本",
            "audio": f"{audio_prefix}/{f5_update60_audio.name}",
            "model": "F5TTS_v1_Base fine-tuned on 500 teacher wavs",
            "size": f"{fmt_bytes(size_bytes(update20_last))} checkpoint / {fmt_bytes(size_bytes(update20_ckpt))} folder",
            "memory": "GB-class; CPU train succeeded, not mobile-suitable",
            "time": "24-27s process / sample on CPU",
            "mobile": "不是手機終點，但可用來判斷老師聲線 transfer 是否變穩",
            "note": "從 update-20 續訓到 update-60；loss 最後約 0.543。這組用來聽音色是否比 update-20 更貼近老師聲線。",
        },
    ]

    extra_samples = [
        ("同文 update-20", "f5_distill500_update20_0002.wav", LINE_TEXT),
        ("新句 A", "f5_distill500_update20_new_a.wav", "我刚刚想了一下，这件事不要急着下结论，我们先把线索整理清楚。"),
        ("新句 B", "f5_distill500_update20_new_b.wav", "欸，你先不要急啦，我有在听，我们一步一步确认就好。"),
        ("同文 update-60", "f5_distill500_update60_0002.wav", LINE_TEXT),
        ("新句 A update-60", "f5_distill500_update60_new_a.wav", "我刚刚想了一下，这件事不要急着下结论，我们先把线索整理清楚。"),
        ("新句 B update-60", "f5_distill500_update60_new_b.wav", "欸，你先不要急啦，我有在听，我们一步一步确认就好。"),
        ("Piper tiny 同文", "piper_tiny_300_0002.wav", LINE_TEXT),
        ("Piper tiny 新句 A", "piper_tiny_300_new_a.wav", "我刚刚想了一下，这件事不要急着下结论，我们先把线索整理清楚。"),
        ("Piper tiny 新句 B", "piper_tiny_300_new_b.wav", "欸，你先不要急啦，我有在听，我们一步一步确认就好。"),
        ("Piper medium 同文", "piper_medium_warm_0002.wav", LINE_TEXT),
        ("Piper medium 新句 A", "piper_medium_warm_new_a.wav", "我刚刚想了一下，这件事不要急着下结论，我们先把线索整理清楚。"),
        ("Piper medium 新句 B", "piper_medium_warm_new_b.wav", "欸，你先不要急啦，我有在听，我们一步一步确认就好。"),
    ]

    cards = []
    for row in rows:
        cards.append(
            f"""
            <article class="card">
              <div class="status">{html.escape(row['status'])}</div>
              <h2>{html.escape(row['name'])}</h2>
              <audio controls preload="metadata" src="{html.escape(row['audio'])}"></audio>
              <dl>
                <div><dt>Model</dt><dd>{html.escape(row['model'])}</dd></div>
                <div><dt>Size</dt><dd>{html.escape(row['size'])}</dd></div>
                <div><dt>Peak memory</dt><dd>{html.escape(row['memory'])}</dd></div>
                <div><dt>Generate time</dt><dd>{html.escape(row['time'])}</dd></div>
                <div><dt>Mobile</dt><dd>{html.escape(row['mobile'])}</dd></div>
              </dl>
              <p>{html.escape(row['note'])}</p>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>taiwan_mandarin_low_r 完整蒸餾實驗 v1</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #171717;
      --muted: #5d6673;
      --line: #d9dee7;
      --paper: #fbfbfd;
      --panel: #ffffff;
      --accent: #b0182b;
      --soft: #f2f6f8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      background: var(--paper);
      color: var(--ink);
      line-height: 1.55;
    }}
    header, main {{ max-width: 1180px; margin: 0 auto; padding: 22px 16px; }}
    header {{ border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 10px; font-size: clamp(25px, 4vw, 40px); letter-spacing: 0; }}
    h2 {{ margin: 4px 0 12px; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 8px 0; }}
    .sub {{ color: var(--muted); max-width: 860px; }}
    .pillrow {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .pill {{ border: 1px solid var(--line); background: var(--panel); border-radius: 999px; padding: 6px 10px; font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; align-items: stretch; }}
    .sample-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .card, .section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .status {{ color: var(--accent); font-size: 13px; font-weight: 700; text-transform: uppercase; }}
    audio {{ width: 100%; height: 44px; margin: 4px 0 10px; }}
    dl {{ margin: 0; display: grid; gap: 7px; }}
    dl div {{ border-top: 1px solid var(--line); padding-top: 7px; }}
    dt {{ color: var(--muted); font-size: 12px; }}
    dd {{ margin: 0; font-weight: 650; overflow-wrap: anywhere; }}
    .section {{ margin-top: 14px; }}
    .steps {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .step {{ background: var(--soft); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .line {{ padding: 10px 12px; border-left: 4px solid var(--accent); background: var(--soft); margin: 14px 0; }}
    a {{ color: var(--accent); }}
    @media (max-width: 860px) {{
      .grid, .steps {{ grid-template-columns: 1fr; }}
      header, main {{ padding-left: 12px; padding-right: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>完整蒸餾實驗 v1</h1>
    <p class="sub">目標聲音：<strong>taiwan_mandarin_low_r</strong>。這頁比較同一句話在老師模型、手機優先 ZipVoice、Piper 小模型蒸餾、以及 F5 訓練分支上的結果。</p>
    <div class="pillrow">
      <span class="pill">Teacher corpus: {stats['ok']}/{stats['rows']} wav</span>
      <span class="pill">Total: {stats['total_seconds']:.1f}s / {stats['total_hours']:.2f}h</span>
      <span class="pill">Avg teacher gen: {stats['avg_generation_seconds']:.2f}s</span>
      <span class="pill">F5 dataset: {fmt_bytes(size_bytes(full_dataset))}</span>
      <span class="pill">Best small student: Piper medium 61MB</span>
    </div>
  </header>
  <main>
    <section class="section">
      <h3>同一句比較</h3>
      <div class="line">{html.escape(LINE_TEXT)}</div>
      <div class="grid">
        {''.join(cards)}
      </div>
    </section>
    <section class="section">
      <h3>500-corpus 額外試聽</h3>
      <div class="sample-list">
        {''.join(f'<article class="card"><div class="status">{html.escape(label)}</div><p>{html.escape(text)}</p><audio controls preload="metadata" src="{html.escape(audio_prefix)}/{html.escape(file)}"></audio></article>' for label, file, text in extra_samples)}
      </div>
    </section>
    <section class="section">
      <h3>現在到底做完了什麼？</h3>
      <div class="steps">
        <div class="step"><strong>1. 老師語料</strong><p>已用 Qwen3 1.7B 產生 {stats['ok']} 條合成老師音訊，約 {stats['total_hours']:.2f} 小時。這是 v1 corpus，文句仍偏模板，還要擴成更自然的 v2。</p></div>
        <div class="step"><strong>2. 資料集</strong><p>已轉成 F5 pinyin dataset，也轉成 Piper/VITS LJSpeech dataset。完整 {stats['ok']} 條可訓練。</p></div>
        <div class="step"><strong>3. 訓練</strong><p>F5 已證明 teacher transfer，但太大。Piper tiny 已做到 14MB ONNX；Piper medium warmstart 已做到 61MB ONNX 且更像。</p></div>
        <div class="step"><strong>4. 手機方向</strong><p>目前最實際是兩條並行：ZipVoice zero-shot 先上 app；Piper medium 繼續完整蒸餾與量化。</p></div>
      </div>
    </section>
    <section class="section">
      <h3>結論</h3>
      <p><strong>已經有真正小模型蒸餾結果。</strong>14MB Piper tiny 證明模型可以非常小，但 from-scratch 300 steps 不夠像；61MB Piper medium warmstart 是目前最合理的小模型分支，proxy score 0.923、同句生成 0.63s、peak RSS 約 266MB。</p>
      <p>下一步要提升像度：用 medium warmstart 繼續訓練更久、擴充 teacher corpus 到 2-5 小時自然句、再做 ONNX int8/float16 量化。ZipVoice 仍是目前最像的手機 baseline，但它是 zero-shot，不是新訓練權重。</p>
      <p><a href="index.html">回三欄 ZipVoice 原報告</a></p>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    html_text = build_html("audio")
    (OUT / "index.html").write_text(html_text, encoding="utf-8")

    zip_audio_dir = ZIPVOICE_REPORT / "audio" / "distillation_v1"
    zip_audio_dir.mkdir(parents=True, exist_ok=True)
    for audio in (OUT / "audio").glob("*.wav"):
        shutil.copy2(audio, zip_audio_dir / audio.name)
    zip_html_text = build_html("audio/distillation_v1")
    (ZIPVOICE_REPORT / "distillation_v1.html").write_text(zip_html_text, encoding="utf-8")
    print(OUT / "index.html")
    print(ZIPVOICE_REPORT / "distillation_v1.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
