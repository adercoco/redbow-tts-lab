#!/usr/bin/env python3
"""Build a standalone report for tiny student 2/160/800 iteration comparison."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
BENCH = BASE / "benchmarks" / "qwen_zipvoice_speedup_v1"
REPORT_DIR = BASE / "reports" / "tiny_zipvoice_student_long800_audition_v1"
REPORT = REPORT_DIR / "tiny_zipvoice_student_long800_audition_v1_standalone.html"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audio_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<audio controls preload="none" src="data:audio/wav;base64,{data}"></audio>'


def fmt_s(value: float) -> str:
    return f"{value:.2f}s"


def mb(value: float) -> str:
    return f"{value:.1f}MB"


def find_row(summary: dict[str, Any], sample_id: str) -> dict[str, Any]:
    for row in summary["rows"]:
        if int(row["step"]) == 3 and row["sample_id"] == sample_id:
            return row
    raise KeyError(sample_id)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    full = read_json(BENCH / "vocoder_quant" / "sherpa_runtime" / "benchmark.json")
    tiny2 = read_json(BENCH / "tiny_student_smoke_runtime" / "benchmark.json")
    tiny160 = read_json(BENCH / "tiny_student_long160_runtime" / "benchmark.json")
    tiny800 = read_json(BENCH / "tiny_student_long800_runtime" / "benchmark.json")

    full3 = full["aggregates"]["sherpa_step3"]
    tiny2_3 = tiny2["aggregates"]["sherpa_step3"]
    tiny160_3 = tiny160["aggregates"]["sherpa_step3"]
    tiny800_3 = tiny800["aggregates"]["sherpa_step3"]

    sample_ids = ["s01", "s02", "s03", "s04", "s05"]
    cards: list[str] = []
    for sid in sample_ids:
        rows = {
            "正常 Qwen→ZipVoice": find_row(full, sid),
            "Tiny 2 iter": find_row(tiny2, sid),
            "Tiny 160 iter": find_row(tiny160, sid),
            "Tiny 800 iter": find_row(tiny800, sid),
        }
        cards.append(
            f"""
            <section class="pair">
              <h3>{html.escape(sid)}・{html.escape(rows["正常 Qwen→ZipVoice"]["text"])}</h3>
              <div class="audios">
                {''.join(f'''
                <div>
                  <b>{html.escape(label)}</b>
                  <small>{fmt_s(row["wall_s"])} / audio {fmt_s(row["audio_s"])}</small>
                  {audio_tag(Path(row["output"]))}
                </div>
                ''' for label, row in rows.items())}
              </div>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tiny ZipVoice 800 iter 長訓試聽</title>
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
    main {{ max-width: 1080px; margin: 0 auto; padding: 24px 16px 56px; }}
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
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
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
    <div class="eyebrow">Tiny ZipVoice 2 → 160 → 800 iter</div>
    <h1>長訓有效，但還不夠好</h1>
    <p class="lead">這次真的跑了長一點：2 iter 幾乎沒講話；160 iter 開始有語音輪廓；800 iter validation loss 降到 0.9067，音長變得更長，但聲音仍偏小、偏糊、節奏不穩。結論：長訓有用，但小架構要好聽，不能只靠這 284 句從零訓。</p>
  </header>

  <section class="summary">
    <div class="box"><b>Tiny 參數</b><div class="value">33.5M</div></div>
    <div class="box"><b>Tiny ONNX 包</b><div class="value">{mb(tiny800["model_size_mb"]["total_mb"])}</div></div>
    <div class="box"><b>2 iter 音長</b><div class="value">{fmt_s(tiny2_3["avg_audio_s"])}</div></div>
    <div class="box"><b>160 iter 音長</b><div class="value">{fmt_s(tiny160_3["avg_audio_s"])}</div></div>
    <div class="box"><b>800 iter 音長</b><div class="value">{fmt_s(tiny800_3["avg_audio_s"])}</div></div>
    <div class="box"><b>800 iter 生成</b><div class="value">{fmt_s(tiny800_3["avg_wall_s"])}</div></div>
  </section>

  <section class="note">
    <p><b>Validation loss：</b>2 iter 3.420；160 iter 1.663；800 iter 0.9067。</p>
    <p><b>我的判斷：</b>這條不是死路，但現在策略不夠。更好的下一步是用 Qwen/ZipVoice 大學生當 teacher，訓練 tiny student 去對齊 teacher 的 flow/feature 目標，或先用更大的合成 corpus pretrain tiny，再做單聲音收斂。只餵 284 句 wav/text 會學到輪廓，但很難學到乾淨、像人的台灣女生聲音。</p>
  </section>

  <h2>同句四欄試聽</h2>
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
