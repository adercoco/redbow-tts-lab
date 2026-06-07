#!/usr/bin/env python3
"""Build a standalone step ladder report for Cosy teacher vs ZipVoice distill."""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = (
    ROOT
    / "distillation"
    / "taiwan_mandarin_low_r"
    / "reports"
    / "cosy_zipvoice_true_distill_step_ladder_v1"
)
OUT_HTML = REPORT_DIR / "cosy_zipvoice_true_distill_step_ladder_v1_standalone.html"

TEACHER_DIR = (
    ROOT / "distillation" / "taiwan_mandarin_low_r" / "teacher_cosy_raw_best2_distill_v1"
)
ZIP_EGS = ROOT / "external" / "ZipVoice" / "egs" / "zipvoice"

LINES = [
    (
        "cosy_01",
        "distill_0001",
        "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
    ),
    (
        "cosy_02",
        "distill_0002",
        "这样听起来有比较像台湾人吗？我希望尾音自然一点，不要太卷舌。",
    ),
    (
        "cosy_03",
        "distill_0003",
        "我想要的不是主播腔，也不是娃娃音，是聪明、温柔、真实的声音。",
    ),
]

VARIANTS = [
    {
        "name": "Cosy teacher",
        "short": "老師",
        "desc": "完整老師聲音，上限參考。",
        "dir": None,
        "size": "4.5GB",
        "peak": "5.73GB RSS",
        "time": "43.00s / 3句",
        "rtf": "約 1.79",
    },
    {
        "name": "ZipVoice 16-step",
        "short": "完整複製",
        "desc": "先看完整 decoding 的複製效果；這是目前聲音上限候選。",
        "dir": ZIP_EGS / "results" / "cosy_true_distill_onnx_int8_step16",
        "size": "175.8MB",
        "peak": "1.55GB RSS",
        "time": "27.61s / 3句",
        "rtf": "1.68",
    },
    {
        "name": "ZipVoice 8-step",
        "short": "折衷",
        "desc": "速度與穩定度折衷；目前比較像可用預設。",
        "dir": ZIP_EGS / "results" / "cosy_true_distill_onnx_int8_step8",
        "size": "175.8MB",
        "peak": "1.54GB RSS",
        "time": "15.44s / 3句",
        "rtf": "0.86",
    },
    {
        "name": "ZipVoice 4-step int8",
        "short": "最快 int8",
        "desc": "速度最快，但你已聽出質感掉比較多。",
        "dir": ZIP_EGS / "results" / "cosy_true_distill_onnx_int8_step4",
        "size": "175.8MB",
        "peak": "1.55GB RSS",
        "time": "9.78s / 3句",
        "rtf": "0.45",
    },
    {
        "name": "ZipVoice 4-step int4 try",
        "short": "int4 測試",
        "desc": "更小但本機變慢；先當實驗，不當推薦。",
        "dir": ZIP_EGS / "results" / "cosy_true_distill_onnx_int4_try_step4",
        "size": "124.1MB",
        "peak": "1.42GB RSS",
        "time": "19.36s / 3句",
        "rtf": "1.08",
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


def variant_audio(variant: dict, zip_id: str, teacher_id: str) -> Path:
    if variant["dir"] is None:
        return TEACHER_DIR / "audio" / f"{teacher_id}.wav"
    return variant["dir"] / f"{zip_id}.wav.wav"


def metric_cards() -> str:
    cards = []
    for item in VARIANTS:
        cards.append(
            f"""
            <article class="metric">
              <div class="metric-name">{html.escape(item["name"])}</div>
              <div class="metric-short">{html.escape(item["short"])}</div>
              <dl>
                <div><dt>大小</dt><dd>{html.escape(item["size"])}</dd></div>
                <div><dt>記憶體</dt><dd>{html.escape(item["peak"])}</dd></div>
                <div><dt>生成</dt><dd>{html.escape(item["time"])}</dd></div>
                <div><dt>RTF</dt><dd>{html.escape(item["rtf"])}</dd></div>
              </dl>
            </article>
            """
        )
    return "\n".join(cards)


def listen_rows() -> str:
    rows = []
    for zip_id, teacher_id, text in LINES:
        cells = []
        for variant in VARIANTS:
            wav = variant_audio(variant, zip_id, teacher_id)
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
              <div class="line-label">同一句比較</div>
              <p class="line-text">{html.escape(text)}</p>
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
  <title>Cosy → ZipVoice 真蒸餾 Step Ladder</title>
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
      width: min(1180px, 100%);
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
      margin: 0;
      max-width: 840px;
      color: var(--muted);
      font-size: 17px;
    }}
    .top-stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .stat, .metric, .audio-card, .callout, .path {{
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
      padding: 14px;
      background: #fff8ed;
    }}
    .good {{ color: var(--good); font-weight: 800; }}
    .warn {{ color: var(--warn); font-weight: 800; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      padding: 12px;
    }}
    .metric-name {{
      font-size: 16px;
      font-weight: 850;
    }}
    .metric-short {{
      color: var(--teal);
      font-size: 13px;
      font-weight: 800;
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
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
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
      margin: 2px 0 12px;
    }}
    .audio-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }}
    .audio-card {{
      padding: 10px;
      background: #fffaf1;
    }}
    .audio-card p {{
      min-height: 62px;
      color: var(--muted);
      font-size: 12px;
    }}
    audio {{
      width: 100%;
      margin-top: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      background: var(--soft);
      color: var(--muted);
      font-size: 12px;
    }}
    tr:last-child td {{ border-bottom: none; }}
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
    @media (max-width: 900px) {{
      main {{ padding: 18px 12px 44px; }}
      .top-stats, .metrics, .audio-grid, .paths {{ grid-template-columns: 1fr; }}
      .audio-card p {{ min-height: 0; }}
      table, thead, tbody, th, td, tr {{ display: block; }}
      thead {{ display: none; }}
      tr {{
        padding: 8px 0;
        border-bottom: 1px solid var(--line);
      }}
      td {{
        border: none;
        padding: 6px 10px;
      }}
      td::before {{
        content: attr(data-label);
        display: block;
        color: var(--muted);
        font-size: 12px;
        font-weight: 800;
      }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Standalone report · {html.escape(generated)} Asia/Taipei</div>
    <h1>Cosy → ZipVoice 真蒸餾 Step Ladder</h1>
    <p class="subtitle">這份報告照你的新方向整理：先聽 16-step 完整複製上限，再慢慢降到 8-step、4-step；另外補上 int4 4-step 實驗，看它是否真的更小更快。</p>
    <div class="top-stats">
      <div class="stat"><span>目前最佳聽感候選</span><strong>先聽 16-step</strong></div>
      <div class="stat"><span>速度折衷候選</span><strong>8-step</strong></div>
      <div class="stat"><span>最快 int8</span><strong>4-step / RTF 0.45</strong></div>
      <div class="stat"><span>int4 結論</span><strong>更小但變慢</strong></div>
    </div>
  </header>

  <section class="band">
    <h2>結論</h2>
    <div class="callout">
      <p><span class="good">新的蒸餾有結果：</span>16-step、8-step、4-step int8 都已可聽；16-step 是目前先確認「完整複製上限」的版本。</p>
      <p><span class="warn">4-step int8 確實比較不穩：</span>你聽到的質感下降是合理的，因為 steps 壓太低，模型還沒有做專門的 few-step teacher training。</p>
      <p><span class="warn">int4 不是目前加速答案：</span>它把核心模型從 175.8MB 降到 124.1MB，peak RSS 從 1.55GB 降到 1.42GB，但三句生成從 9.78s 變成 19.36s。</p>
    </div>
  </section>

  <section class="band">
    <h2>關鍵數字</h2>
    <div class="metrics">{metric_cards()}</div>
  </section>

  <section class="band">
    <h2>速度階梯</h2>
    <table>
      <thead><tr><th>版本</th><th>模型核心大小</th><th>Peak RSS</th><th>三句生成</th><th>平均 RTF</th><th>判斷</th></tr></thead>
      <tbody>
        <tr><td data-label="版本">Cosy teacher</td><td data-label="模型核心大小">4.5GB</td><td data-label="Peak RSS">5.73GB</td><td data-label="三句生成">43.00s</td><td data-label="平均 RTF">約 1.79</td><td data-label="判斷">聲音上限，不適合手機。</td></tr>
        <tr><td data-label="版本">ZipVoice 16-step int8</td><td data-label="模型核心大小">175.8MB</td><td data-label="Peak RSS">1.55GB</td><td data-label="三句生成">27.61s</td><td data-label="平均 RTF">1.68</td><td data-label="判斷">先聽完整複製效果。</td></tr>
        <tr><td data-label="版本">ZipVoice 8-step int8</td><td data-label="模型核心大小">175.8MB</td><td data-label="Peak RSS">1.54GB</td><td data-label="三句生成">15.44s</td><td data-label="平均 RTF">0.86</td><td data-label="判斷">比較像當前可用折衷。</td></tr>
        <tr><td data-label="版本">ZipVoice 4-step int8</td><td data-label="模型核心大小">175.8MB</td><td data-label="Peak RSS">1.55GB</td><td data-label="三句生成">9.78s</td><td data-label="平均 RTF">0.45</td><td data-label="判斷">最快，但音質目前掉。</td></tr>
        <tr><td data-label="版本">ZipVoice 4-step int4 try</td><td data-label="模型核心大小">124.1MB</td><td data-label="Peak RSS">1.42GB</td><td data-label="三句生成">19.36s</td><td data-label="平均 RTF">1.08</td><td data-label="判斷">更小但變慢，不推薦當速度版。</td></tr>
      </tbody>
    </table>
  </section>

  <section class="band">
    <h2>試聽：先完整，再降階</h2>
    <p>每一句按順序聽：Cosy teacher → 16-step → 8-step → 4-step int8 → 4-step int4。重點找出從哪一階開始不像、糊、掉字、尾音怪。</p>
    {listen_rows()}
  </section>

  <section class="band">
    <h2>為什麼 4-step 會掉</h2>
    <p>ZipVoice 是 flow-matching 類模型，steps 代表生成時從雜訊走到語音的修正次數。16-step 給模型比較多機會修正音色、咬字和韻律；4-step 是硬把路徑縮短，速度會快，但沒有做專門 few-step distillation 時，聲音比較容易粗、糊、少了老師的細節。</p>
    <p>所以正確路線不是硬用 4-step，而是先用 16-step 確認「有沒有學像」，再訓練 8/4-step 的 few-step student，讓低 steps 也學到 16-step 的輸出分布。</p>
  </section>

  <section class="band">
    <h2>手機即時的下一步</h2>
    <div class="callout">
      <p><strong>短期：</strong>用 8-step 當 demo 預設，4-step 當快速模式。先不要把 4-step 當品質版。</p>
      <p><strong>中期：</strong>做 16-step teacher → 4-step student 的 few-step distillation，目標是讓 4-step 聽起來接近 8/16-step。</p>
      <p><strong>長期：</strong>要真正在中階手機即時，還需要原生 sherpa-onnx mobile benchmark、切句 streaming、vocoder 最佳化，以及更小架構學生。int4 只有在手機 runtime 有高效 4-bit kernel 時才值得繼續。</p>
    </div>
  </section>

  <section class="band">
    <h2>檔案位置</h2>
    <div class="paths">
      <div class="path"><strong>本報告</strong><br><code>{html.escape(rel(OUT_HTML))}</code></div>
      <div class="path"><strong>16-step outputs</strong><br><code>{html.escape(rel(ZIP_EGS / "results" / "cosy_true_distill_onnx_int8_step16"))}</code></div>
      <div class="path"><strong>8-step outputs</strong><br><code>{html.escape(rel(ZIP_EGS / "results" / "cosy_true_distill_onnx_int8_step8"))}</code></div>
      <div class="path"><strong>4-step int8 outputs</strong><br><code>{html.escape(rel(ZIP_EGS / "results" / "cosy_true_distill_onnx_int8_step4"))}</code></div>
      <div class="path"><strong>4-step int4 outputs</strong><br><code>{html.escape(rel(ZIP_EGS / "results" / "cosy_true_distill_onnx_int4_try_step4"))}</code></div>
      <div class="path"><strong>ONNX int4 try model</strong><br><code>{html.escape(rel(ZIP_EGS / "exp" / "zipvoice_cosy_teacher_smoke_onnx_ckpt60_int4_try"))}</code></div>
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
