#!/usr/bin/env python3
"""Build a standalone audition page for the ZipVoice student checkpoint."""

from __future__ import annotations

import base64
import html
import json
import re
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
REPORT_DIR = BASE / "reports" / "zipvoice_student_600_audition_v1"
OUT = REPORT_DIR / "zipvoice_student_600_audition_v1_standalone.html"
MODEL_ONLY = (
    ROOT
    / "external"
    / "ZipVoice"
    / "egs"
    / "zipvoice"
    / "exp"
    / "zipvoice_cosy_golden_daily_500_decoder_v1"
    / "checkpoint-600-model-only.pt"
)
FULL_CHECKPOINT = MODEL_ONLY.with_name("checkpoint-600.pt")
REFERENCE = (
    BASE
    / "golden_teacher"
    / "cosyvoice2_clear_best2_line04_v1"
    / "reference_clear_best2_7s.wav"
)
SAME_TEXT_TEACHER = (
    REPORT_DIR
    / "cosy_same_text_teacher"
    / "audio"
    / "same_text_s03.wav"
)
SAME_TEXT = "你先不要急，我们慢慢来，把事情一件一件处理好。"


TEXTS = [
    ("s01", "等一下我先看一下讯息，晚一点再跟你说。"),
    ("s02", "今天有点累，不过听到你这样说，我心情有好一点。"),
    ("s03", "你先不要急，我们慢慢来，把事情一件一件处理好。"),
]
STEPS = [16, 8, 4]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def data_uri(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def parse_step_log(step: int) -> dict[str, float]:
    log = REPORT_DIR / "logs" / f"step{step}.log"
    values: dict[str, float] = {}
    if not log.exists():
        return values
    text = log.read_text(encoding="utf-8", errors="replace")
    for idx, rtf in re.findall(r"\[Sentence: (\d+)\] RTF: ([0-9.]+)", text):
        values[f"s{int(idx) + 1:02d}"] = float(rtf)
    avg = re.search(r"Average RTF: ([0-9.]+)", text)
    if avg:
        values["avg"] = float(avg.group(1))
    return values


def audio_card(step: int, sample_id: str, text: str, rtf: float | None) -> str:
    path = REPORT_DIR / "audio" / f"step{step}" / f"zip_student_{sample_id}.wav"
    if not path.exists():
        return f"<td data-label='{step}-step'>missing</td>"
    seconds = wav_seconds(path)
    gen_seconds = rtf * seconds if rtf is not None else None
    meta = f"{seconds:.2f}s audio"
    if gen_seconds is not None:
        meta += f" / generated {gen_seconds:.2f}s / RTF {rtf:.2f}"
    return f"""
      <td data-label="{step}-step">
        <audio controls preload="metadata" src="{data_uri(path)}"></audio>
        <div class="meta">{esc(meta)}</div>
      </td>
    """


def main() -> int:
    rtfs = {step: parse_step_log(step) for step in STEPS}
    avg_cells = "".join(
        f"<div class='metric'><b>{rtfs[step].get('avg', 0):.2f}</b><span>{step}-step avg RTF</span></div>"
        for step in STEPS
    )
    rows = []
    for sample_id, text in TEXTS:
        cells = "".join(
            audio_card(step, sample_id, text, rtfs[step].get(sample_id))
            for step in STEPS
        )
        rows.append(
            f"""
            <tr>
              <td data-label="Text"><b>{esc(sample_id)}</b><p>{esc(text)}</p></td>
              {cells}
            </tr>
            """
        )

    summary = {
        "checkpoint": str(MODEL_ONLY),
        "checkpoint_size_mb": round(size_mb(MODEL_ONLY), 1),
        "full_checkpoint_size_mb": round(size_mb(FULL_CHECKPOINT), 1),
        "avg_rtf": {str(step): rtfs[step].get("avg") for step in STEPS},
    }
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ZipVoice Student 600 試聽</title>
  <style>
    :root {{
      --bg: #f6f1e9;
      --paper: #fffdf8;
      --ink: #25211c;
      --muted: #6b6259;
      --line: #ded2c2;
      --red: #b33a3a;
      --blue: #245f83;
      --green: #407058;
      --soft: #f0e4d6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      font-size: 16px;
      line-height: 1.6;
    }}
    main {{
      width: min(980px, 100%);
      margin: 0 auto;
      padding: 20px 14px 48px;
    }}
    header {{ padding: 20px 0 12px; }}
    .eyebrow {{ color: var(--red); font-size: 13px; font-weight: 800; }}
    h1 {{
      margin: 6px 0 10px;
      font-size: clamp(30px, 8vw, 46px);
      line-height: 1.08;
      letter-spacing: 0;
    }}
    h2 {{ margin: 0 0 10px; font-size: 22px; }}
    p {{ margin: 7px 0; }}
    section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 14px 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .metric {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .metric b {{
      display: block;
      color: var(--blue);
      font-size: 25px;
      line-height: 1.1;
    }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    .meta {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
    audio {{ width: 100%; min-width: 180px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      vertical-align: top;
    }}
    th {{ color: var(--muted); text-align: left; }}
    code {{
      background: #eee4d8;
      border-radius: 5px;
      padding: 2px 5px;
      word-break: break-word;
    }}
    .path {{ color: var(--muted); word-break: break-word; font-size: 13px; }}
    .refbox {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .refitem {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    @media (max-width: 720px) {{
      table, thead, tbody, tr, th, td {{ display: block; }}
      thead {{ display: none; }}
      tr {{ border-bottom: 1px solid var(--line); padding: 8px 0; }}
      td {{ border-bottom: 0; padding: 8px 0; }}
      td::before {{
        content: attr(data-label);
        display: block;
        color: var(--muted);
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 3px;
      }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">ZipVoice Student · checkpoint 600 · 2026-06-05</div>
    <h1>ZipVoice 蒸餾版試聽</h1>
    <p>這頁是剛訓完的 ZipVoice student，使用 <code>checkpoint-600-model-only.pt</code>。音檔標示的文字都對齊實際內容，避免 reference 聽起來和句子不一致。</p>
  </header>

  <section>
    <h2>模型與速度</h2>
    <div class="grid">
      <div class="metric"><b>{size_mb(MODEL_ONLY):.0f} MB</b><span>model-only checkpoint</span></div>
      <div class="metric"><b>{size_mb(FULL_CHECKPOINT):.0f} MB</b><span>full training checkpoint</span></div>
      {avg_cells}
    </div>
    <p class="path">Model: {esc(MODEL_ONLY)}</p>
    <p>RTF 越低越快。這裡是 Mac PyTorch CPU 測試，不是手機 ONNX/int8 最終速度。</p>
  </section>

  <section>
    <h2>Reference</h2>
    <div class="refbox">
      <div class="refitem">
        <b>Cosy teacher，同一句</b>
        <p>{esc(SAME_TEXT)}</p>
        <audio controls preload="metadata" src="{data_uri(SAME_TEXT_TEACHER)}"></audio>
      </div>
      <div class="refitem">
        <b>Prompt reference</b>
        <p>你覺得比較好的聲音錨點；內容不是測試句，只用來引導音色。</p>
        <audio controls preload="metadata" src="{data_uri(REFERENCE)}"></audio>
      </div>
    </div>
  </section>

  <section>
    <h2>ZipVoice Student 試聽</h2>
    <table>
      <thead>
        <tr>
          <th>Text</th>
          <th>16-step</th>
          <th>8-step</th>
          <th>4-step</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </section>
</main>
</body>
</html>
"""
    OUT.write_text(html_text, encoding="utf-8")
    print(OUT)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
