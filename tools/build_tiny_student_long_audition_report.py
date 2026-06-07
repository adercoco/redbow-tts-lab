#!/usr/bin/env python3
"""Build a standalone audition report comparing tiny student training lengths."""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
BENCH = BASE / "benchmarks" / "qwen_zipvoice_speedup_v1"
REPORT_DIR = BASE / "reports" / "tiny_zipvoice_student_long160_audition_v1"
REPORT = REPORT_DIR / "tiny_zipvoice_student_long160_audition_v1_standalone.html"
LOG = (
    ROOT
    / "external"
    / "ZipVoice"
    / "egs"
    / "zipvoice"
    / "exp"
    / "zipvoice_mobile_student_qwen_tiny_long160_cdr0_lr5e4"
    / "log"
    / "log-train-2026-06-05-11-27-20"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audio_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<audio controls preload="none" src="data:audio/wav;base64,{data}"></audio>'


def fmt_s(value: float) -> str:
    return f"{value:.2f}s"


def mb(value: float) -> str:
    return f"{value:.1f}MB"


def find_row(summary: dict[str, Any], step: int, sample_id: str) -> dict[str, Any]:
    for row in summary["rows"]:
        if int(row["step"]) == step and row["sample_id"] == sample_id:
            return row
    raise KeyError((step, sample_id))


def parse_losses() -> list[tuple[int, float]]:
    losses: list[tuple[int, float]] = []
    pattern = re.compile(r"global_batch_idx: (\\d+), validation: loss=([0-9.]+)")
    for line in LOG.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            losses.append((int(match.group(1)), float(match.group(2))))
    return losses


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    full = read_json(BENCH / "vocoder_quant" / "sherpa_runtime" / "benchmark.json")
    tiny2 = read_json(BENCH / "tiny_student_smoke_runtime" / "benchmark.json")
    tiny160 = read_json(BENCH / "tiny_student_long160_runtime" / "benchmark.json")

    full3 = full["aggregates"]["sherpa_step3"]
    tiny2_3 = tiny2["aggregates"]["sherpa_step3"]
    tiny160_3 = tiny160["aggregates"]["sherpa_step3"]
    losses = parse_losses()
    loss_line = " → ".join(f"{idx}: {loss:.3f}" for idx, loss in losses)

    sample_ids = ["s01", "s02", "s03", "s04", "s05"]
    cards: list[str] = []
    for sid in sample_ids:
        full_row = find_row(full, 3, sid)
        tiny2_row = find_row(tiny2, 3, sid)
        tiny160_row = find_row(tiny160, 3, sid)
        cards.append(
            f"""
            <section class="pair">
              <h3>{html.escape(sid)}・{html.escape(full_row["text"])}</h3>
              <div class="audios">
                <div>
                  <b>正常 Qwen→ZipVoice</b>
                  <small>{fmt_s(full_row["wall_s"])} / audio {fmt_s(full_row["audio_s"])}</small>
                  {audio_tag(Path(full_row["output"]))}
                </div>
                <div>
                  <b>Tiny 2 iter</b>
                  <small>{fmt_s(tiny2_row["wall_s"])} / audio {fmt_s(tiny2_row["audio_s"])}</small>
                  {audio_tag(Path(tiny2_row["output"]))}
                </div>
                <div>
                  <b>Tiny 160 iter</b>
                  <small>{fmt_s(tiny160_row["wall_s"])} / audio {fmt_s(tiny160_row["audio_s"])}</small>
                  {audio_tag(Path(tiny160_row["output"]))}
                </div>
              </div>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tiny ZipVoice 160 iter 試聽</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #fafafa;
      color: #202124;
      font-family: -apple-system, BlinkMacSystemFont, "Noto Sans TC", "PingFang TC", "Segoe UI", sans-serif;
      line-height: 1.6;
      font-size: 16px;
    }}
    main {{ max-width: 980px; margin: 0 auto; padding: 24px 16px 52px; }}
    header {{ border-bottom: 1px solid #e5e7eb; padding-bottom: 16px; }}
    .eyebrow {{ color: #b91c1c; font-weight: 800; font-size: 13px; }}
    h1 {{ font-size: clamp(28px, 8vw, 42px); line-height: 1.12; margin: 8px 0 10px; letter-spacing: 0; }}
    h2 {{ font-size: 22px; margin: 30px 0 12px; letter-spacing: 0; }}
    h3 {{ font-size: 17px; line-height: 1.35; margin: 0 0 12px; letter-spacing: 0; }}
    p {{ margin: 0 0 12px; }}
    .lead {{ color: #667085; font-size: 17px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .box, .pair, .note {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 14px;
    }}
    .box b {{ display: block; color: #667085; font-size: 13px; }}
    .value {{ font-size: 26px; font-weight: 850; margin-top: 3px; }}
    .note {{ border-left: 4px solid #b91c1c; }}
    .audios {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .audios > div {{
      background: #f7f7f4;
      border-radius: 8px;
      padding: 12px;
    }}
    small {{ display: block; color: #667085; margin: 4px 0 8px; }}
    audio {{ width: 100%; }}
    code {{ background: #f2f4f7; padding: 1px 4px; border-radius: 5px; }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Tiny ZipVoice 160 iter experiment</div>
    <h1>長訓有沒有救 tiny？</h1>
    <p class="lead">短答：有救到「開始像音訊」，但還沒有救到「可用 TTS」。2 iter 幾乎沒講話；160 iter 已經變長、有聲音、有能量，但音量/清晰度/發音仍離正常 Qwen→ZipVoice 很遠。</p>
  </header>

  <section class="summary">
    <div class="box"><b>Tiny 參數</b><div class="value">33.5M</div></div>
    <div class="box"><b>Tiny ONNX 包</b><div class="value">{mb(tiny160["model_size_mb"]["total_mb"])}</div></div>
    <div class="box"><b>2 iter 平均音長</b><div class="value">{fmt_s(tiny2_3["avg_audio_s"])}</div></div>
    <div class="box"><b>160 iter 平均音長</b><div class="value">{fmt_s(tiny160_3["avg_audio_s"])}</div></div>
    <div class="box"><b>160 iter 生成</b><div class="value">{fmt_s(tiny160_3["avg_wall_s"])}</div></div>
  </section>

  <section class="note">
    <p><b>Validation loss：</b>{html.escape(loss_line)}</p>
    <p><b>我的判斷：</b>長訓是有用的，因為 loss 和 duration 都明顯改善；但只靠 284 句從零訓 33.5M tiny，還不足以得到好聲音。下一步應該跑 1000-3000 iter，或更好：用完整 ZipVoice teacher 的 feature/flow 目標做真正 student distillation，而不是只用 wav/text 從零學。</p>
  </section>

  <h2>同句三欄試聽</h2>
  {"".join(cards)}
</main>
</body>
</html>
"""
    REPORT.write_text(html_text, encoding="utf-8")
    print(REPORT)
    print(f"size_mb={REPORT.stat().st_size / 1024 / 1024:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
