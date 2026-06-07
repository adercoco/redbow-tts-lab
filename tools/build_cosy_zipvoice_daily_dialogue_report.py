#!/usr/bin/env python3
"""Build a standalone daily dialogue report for Cosy vs ZipVoice distillation."""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = (
    ROOT
    / "distillation"
    / "taiwan_mandarin_low_r"
    / "reports"
    / "cosy_zipvoice_daily_dialogue_v1"
)
OUT_HTML = REPORT_DIR / "cosy_zipvoice_daily_dialogue_v1_standalone.html"

TEACHER_DIR = ROOT / "distillation" / "taiwan_mandarin_low_r" / "teacher_cosy_daily_dialogue_v1"
ZIP_EGS = ROOT / "external" / "ZipVoice" / "egs" / "zipvoice"

VARIANTS = [
    {
        "name": "Cosy teacher",
        "desc": "老師目標聲音，正常日常句。",
        "dir": None,
        "size": "4.5GB",
        "peak": "5.73GB RSS；沿用同模型量測",
        "time": "33.44s / 4句；本批 Cosy 生成耗時合計",
        "rtf": "約 1.70",
    },
    {
        "name": "ZipVoice 16-step",
        "desc": "完整複製上限，先判斷像不像。",
        "dir": ZIP_EGS / "results" / "cosy_daily_dialogue_true_distill_onnx_int8_step16",
        "size": "175.8MB",
        "peak": "1.34GB RSS",
        "time": "36.11s / 4句",
        "rtf": "2.09",
    },
    {
        "name": "ZipVoice 8-step",
        "desc": "品質與速度折衷。",
        "dir": ZIP_EGS / "results" / "cosy_daily_dialogue_true_distill_onnx_int8_step8",
        "size": "175.8MB",
        "peak": "1.34GB RSS",
        "time": "19.56s / 4句",
        "rtf": "1.07",
    },
    {
        "name": "ZipVoice 4-step",
        "desc": "最快 int8 版，檢查是否可接受。",
        "dir": ZIP_EGS / "results" / "cosy_daily_dialogue_true_distill_onnx_int8_step4",
        "size": "175.8MB",
        "peak": "1.34GB RSS",
        "time": "11.43s / 4句",
        "rtf": "0.55",
    },
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def audio_src(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def audio_tag(path: Path) -> str:
    if not path.exists():
        return '<div class="missing">missing audio</div>'
    return f'<audio controls preload="none" src="{audio_src(path)}"></audio>'


def load_rows() -> list[dict]:
    return json.loads((TEACHER_DIR / "manifest.json").read_text(encoding="utf-8"))


def variant_audio(variant: dict, row: dict) -> Path:
    if variant["dir"] is None:
        return Path(row["audio"])
    return variant["dir"] / f"{row['id']}.wav.wav"


def metric_cards() -> str:
    cards = []
    for variant in VARIANTS:
        cards.append(
            f"""
            <article class="metric">
              <div class="metric-name">{html.escape(variant["name"])}</div>
              <dl>
                <div><dt>大小</dt><dd>{html.escape(variant["size"])}</dd></div>
                <div><dt>記憶體</dt><dd>{html.escape(variant["peak"])}</dd></div>
                <div><dt>生成時間</dt><dd>{html.escape(variant["time"])}</dd></div>
                <div><dt>RTF</dt><dd>{html.escape(variant["rtf"])}</dd></div>
              </dl>
            </article>
            """
        )
    return "\n".join(cards)


def listen_rows() -> str:
    rows = []
    for row in load_rows():
        cells = []
        for variant in VARIANTS:
            wav = variant_audio(variant, row)
            cells.append(
                f"""
                <article class="audio-card">
                  <h3>{html.escape(variant["name"])}</h3>
                  <p>{html.escape(variant["desc"])}</p>
                  {audio_tag(wav)}
                </article>
                """
            )
        rows.append(
            f"""
            <section class="listen-row">
              <div class="line-label">日常對話</div>
              <p class="line-text">{html.escape(row["text"])}</p>
              <div class="line-meta">Cosy teacher generation: {row["seconds"]:.2f}s</div>
              <div class="audio-grid">{''.join(cells)}</div>
            </section>
            """
        )
    return "\n".join(rows)


def build() -> str:
    generated = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cosy → ZipVoice 日常對話蒸餾對比</title>
  <style>
    :root {{
      --bg: #f6f2eb;
      --paper: #fffdf8;
      --ink: #201b17;
      --muted: #71665d;
      --line: #ded3c5;
      --accent: #aa3d2b;
      --teal: #265c63;
      --soft: #eee5d8;
      --good: #2c6d4d;
      --warn: #9a5d1c;
      --radius: 8px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", "PingFang TC", sans-serif;
      line-height: 1.6;
      letter-spacing: 0;
    }}
    main {{
      width: min(1120px, 100%);
      margin: 0 auto;
      padding: 22px 14px 54px;
    }}
    header {{
      padding: 22px 0 18px;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 800;
    }}
    h1 {{
      margin: 8px 0 10px;
      font-size: clamp(30px, 6vw, 54px);
      line-height: 1.06;
      letter-spacing: 0;
    }}
    .subtitle {{
      max-width: 820px;
      color: var(--muted);
      font-size: 17px;
      margin: 0;
    }}
    .top-stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .stat, .metric, .listen-row, .audio-card, .callout, .path {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
    }}
    .stat {{
      padding: 12px;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}
    .stat strong {{
      display: block;
      margin-top: 4px;
      font-size: 20px;
      line-height: 1.15;
    }}
    section.band {{
      padding: 24px 0;
      border-bottom: 1px solid var(--line);
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 24px;
      line-height: 1.22;
    }}
    h3 {{
      margin: 0 0 6px;
      font-size: 16px;
      line-height: 1.25;
    }}
    p {{ margin: 0 0 10px; }}
    .callout {{
      border-left: 4px solid var(--accent);
      border-radius: 0 var(--radius) var(--radius) 0;
      background: #fff8ed;
      padding: 14px;
    }}
    .good {{ color: var(--good); font-weight: 800; }}
    .warn {{ color: var(--warn); font-weight: 800; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{ padding: 12px; }}
    .metric-name {{
      font-size: 17px;
      font-weight: 850;
      margin-bottom: 8px;
    }}
    dl {{
      margin: 0;
      display: grid;
      gap: 7px;
    }}
    dl div {{
      border-top: 1px solid var(--line);
      padding-top: 7px;
    }}
    dt {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}
    dd {{
      margin: 1px 0 0;
      font-size: 14px;
      font-weight: 650;
    }}
    .listen-row {{
      padding: 14px;
      margin-bottom: 12px;
    }}
    .line-label {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 850;
    }}
    .line-text {{
      font-size: 17px;
      font-weight: 780;
      margin: 2px 0;
    }}
    .line-meta {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
    }}
    .audio-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .audio-card {{
      padding: 10px;
      background: #fffaf1;
    }}
    .audio-card p {{
      min-height: 44px;
      color: var(--muted);
      font-size: 12px;
    }}
    audio {{
      width: 100%;
      margin-top: 8px;
    }}
    .paths {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .path {{
      padding: 10px;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 860px) {{
      main {{ padding: 18px 12px 44px; }}
      .top-stats, .metrics, .audio-grid, .paths {{ grid-template-columns: 1fr; }}
      .audio-card p {{ min-height: 0; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Standalone report · {html.escape(generated)} Asia/Taipei</div>
    <h1>Cosy → ZipVoice 日常對話蒸餾對比</h1>
    <p class="subtitle">這版把測試文本改成正常人日常對話：等人、買飲料、晚餐、回訊息。先聽 Cosy teacher，再聽 ZipVoice 16 / 8 / 4-step。</p>
    <div class="top-stats">
      <div class="stat"><span>文本風格</span><strong>日常對話</strong></div>
      <div class="stat"><span>完整複製</span><strong>16-step</strong></div>
      <div class="stat"><span>折衷</span><strong>8-step</strong></div>
      <div class="stat"><span>最快</span><strong>4-step</strong></div>
    </div>
  </header>

  <section class="band">
    <h2>這版改了什麼</h2>
    <div class="callout">
      <p><span class="good">文本換成日常句：</span>不再用查線索、證明自己那種戲劇句，改成一般人比較會講的生活對話。</p>
      <p><span class="warn">速度會因句子不同而變：</span>這批日常句的 16-step 比上一批慢一點，平均 RTF `2.09`；4-step 平均 RTF `0.55`，還是最快。</p>
    </div>
  </section>

  <section class="band">
    <h2>關鍵資訊</h2>
    <div class="metrics">{metric_cards()}</div>
  </section>

  <section class="band">
    <h2>試聽：日常對話</h2>
    <p>每一句按順序聽：Cosy teacher → 16-step → 8-step → 4-step。這樣比較能判斷真正 app 裡日常說話會不會自然。</p>
    {listen_rows()}
  </section>

  <section class="band">
    <h2>檔案位置</h2>
    <div class="paths">
      <div class="path"><strong>本報告</strong><br><code>{html.escape(rel(OUT_HTML))}</code></div>
      <div class="path"><strong>Cosy daily teacher</strong><br><code>{html.escape(rel(TEACHER_DIR))}</code></div>
      <div class="path"><strong>ZipVoice daily TSV</strong><br><code>{html.escape(rel(ZIP_EGS / "test_cosy_daily_dialogue.tsv"))}</code></div>
      <div class="path"><strong>4-step outputs</strong><br><code>{html.escape(rel(ZIP_EGS / "results" / "cosy_daily_dialogue_true_distill_onnx_int8_step4"))}</code></div>
    </div>
  </section>
</main>
</body>
</html>
"""


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build(), encoding="utf-8")
    print(OUT_HTML)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
