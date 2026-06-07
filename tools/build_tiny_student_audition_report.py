#!/usr/bin/env python3
"""Build a standalone audition report for the tiny ZipVoice student smoke model."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
BENCH = BASE / "benchmarks" / "qwen_zipvoice_speedup_v1"
REPORT_DIR = BASE / "reports" / "tiny_zipvoice_student_smoke_audition_v1"
REPORT = REPORT_DIR / "tiny_zipvoice_student_smoke_audition_v1_standalone.html"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audio_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<audio controls preload="none" src="data:audio/wav;base64,{data}"></audio>'


def fmt_s(value: float) -> str:
    return f"{value:.2f}s"


def mb(value: float) -> str:
    return f"{value:.1f}MB"


def find_row(rows: list[dict[str, Any]], step: int, sample_id: str) -> dict[str, Any]:
    for row in rows:
        if int(row["step"]) == step and row["sample_id"] == sample_id:
            return row
    raise KeyError((step, sample_id))


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    full = read_json(BENCH / "vocoder_quant" / "sherpa_runtime" / "benchmark.json")
    tiny = read_json(BENCH / "tiny_student_smoke_runtime" / "benchmark.json")

    full_agg = full["aggregates"]["sherpa_step3"]
    tiny_agg = tiny["aggregates"]["sherpa_step3"]
    sample_ids = ["s01", "s02", "s03", "s04", "s05"]

    cards = []
    for sample_id in sample_ids:
        full_row = find_row(full["rows"], 3, sample_id)
        tiny_row = find_row(tiny["rows"], 3, sample_id)
        cards.append(
            f"""
            <section class="pair">
              <h3>{html.escape(sample_id)}・{html.escape(full_row["text"])}</h3>
              <div class="audios">
                <div>
                  <b>正常 Qwen→ZipVoice 3-step</b>
                  <small>{fmt_s(full_row["wall_s"])} / audio {fmt_s(full_row["audio_s"])}</small>
                  {audio_tag(Path(full_row["output"]))}
                </div>
                <div>
                  <b>Tiny student smoke 3-step</b>
                  <small>{fmt_s(tiny_row["wall_s"])} / audio {fmt_s(tiny_row["audio_s"])}</small>
                  {audio_tag(Path(tiny_row["output"]))}
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
  <title>Tiny ZipVoice 小架構試聽</title>
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
    main {{ max-width: 860px; margin: 0 auto; padding: 24px 16px 52px; }}
    header {{ border-bottom: 1px solid #e5e7eb; padding-bottom: 16px; }}
    .eyebrow {{ color: #b91c1c; font-weight: 800; font-size: 13px; }}
    h1 {{ font-size: clamp(28px, 8vw, 42px); line-height: 1.12; margin: 8px 0 10px; letter-spacing: 0; }}
    h2 {{ font-size: 22px; margin: 30px 0 12px; letter-spacing: 0; }}
    h3 {{ font-size: 17px; line-height: 1.35; margin: 0 0 12px; letter-spacing: 0; }}
    p {{ margin: 0 0 12px; }}
    .lead {{ color: #667085; font-size: 17px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .box, .pair {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 14px;
    }}
    .box b {{ display: block; color: #667085; font-size: 13px; }}
    .value {{ font-size: 28px; font-weight: 850; margin-top: 3px; }}
    .warn {{
      border-left: 4px solid #b91c1c;
      background: white;
      border-radius: 6px;
      padding: 12px 14px;
    }}
    .audios {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
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
    <div class="eyebrow">Tiny ZipVoice student smoke audition</div>
    <h1>小架構效果試聽</h1>
    <p class="lead">這不是成品聲音。它是 33.5M 參數 tiny ZipVoice，只訓練 2 iteration，用來確認小模型能不能進 ZipVoice → ONNX → sherpa pipeline。你會聽到它現在還沒學會正常 duration/音色，但可以拿來判斷這條路的初始狀態。</p>
  </header>

  <section class="summary">
    <div class="box"><b>正常包大小</b><div class="value">{mb(full["model_size_mb"]["total_mb"])}</div></div>
    <div class="box"><b>Tiny 包大小</b><div class="value">{mb(tiny["model_size_mb"]["total_mb"])}</div></div>
    <div class="box"><b>正常平均一句</b><div class="value">{fmt_s(full_agg["avg_wall_s"])}</div></div>
    <div class="box"><b>Tiny smoke 平均一句</b><div class="value">{fmt_s(tiny_agg["avg_wall_s"])}</div></div>
  </section>

  <section class="warn">
    <p><b>白話判斷：</b>小架構目前「有跑起來」，但還「不會講好」。如果要聽起來接近 Qwen 老師，下一步不是再調推論，而是要長訓這個 33.5M student，再做 4/3-step distillation。</p>
  </section>

  <h2>同句對照</h2>
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
