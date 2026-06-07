#!/usr/bin/env python3
import base64
import html
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import wavfile


ROOT = Path("/Users/ader/Documents/App")
OUT_DIR = ROOT / "distillation/taiwan_mandarin_low_r/reports/mobile_standalone_audio_v1"
OUT_HTML = OUT_DIR / "mobile_standalone_audio_v1.html"


ENTRIES = [
    {
        "group": "目前手機 App 版",
        "title": "Matcha app 最新慢速大聲版",
        "text": "你先不要急，我们慢慢来，把事情一件一件处理好。",
        "path": ROOT / "distillation/taiwan_mandarin_low_r/reports/redbow_app_audio_debug/matcha_ios_sim_slow_loud_selftest.wav",
        "note": "現在紅色蝴蝶結 app 的主路線：Matcha-CosyGolden + HiFi-GAN T2，speed 0.92，iOS runtime 有音量、濾波、limiter 後處理。",
        "badge": "App current",
    },
    {
        "group": "目前手機 App 版",
        "title": "Matcha app 前一版",
        "text": "你先不要急，我们慢慢来，把事情一件一件处理好。",
        "path": ROOT / "distillation/taiwan_mandarin_low_r/reports/redbow_app_audio_debug/matcha_ios_sim_selftest.wav",
        "note": "前一版自測輸出，用來比較變慢、變大聲、後處理後有沒有比較自然清楚。",
        "badge": "Before tune",
    },
    {
        "group": "Cosy / ZipVoice 對照",
        "title": "CosyVoice2 golden teacher",
        "text": "如果你愿意的话，我们等一下再一起确认一次。",
        "path": ROOT / "distillation/taiwan_mandarin_low_r/golden_teacher/cosyvoice2_clear_best2_line04_v1/golden_teacher.wav",
        "note": "你選定的 golden teacher 聲音錨點。後續 Cosy -> ZipVoice / Matcha 都以這個音色方向為目標。",
        "badge": "Teacher",
    },
    {
        "group": "Cosy / ZipVoice 對照",
        "title": "Cosy teacher：句子 1",
        "text": "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
        "path": ROOT / "distillation/taiwan_mandarin_low_r/reports/zipvoice_4step_optimize_v1/assets/teacher/cosy_01.wav",
        "note": "ZipVoice speed report 裡的 Cosy 老師基準。",
        "badge": "Teacher",
    },
    {
        "group": "Cosy / ZipVoice 對照",
        "title": "ZipVoice qint8 Vocos + mild clean：句子 1",
        "text": "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
        "path": ROOT / "distillation/taiwan_mandarin_low_r/reports/zipvoice_4step_optimize_v1/assets/qvocoder_clean_mild/cosy_01.wav",
        "note": "你覺得還不錯的 ZipVoice 4-step + qint8 Vocos + 保守清理版。這條路仍有電子雜訊風險，所以目前 app 先放 Matcha。",
        "badge": "ZipVoice",
    },
    {
        "group": "Cosy / ZipVoice 對照",
        "title": "ZipVoice qint8 Vocos + mild clean：句子 2",
        "text": "这样听起来有比较像台湾人吗？我希望尾音自然一点，不要太卷舌。",
        "path": ROOT / "distillation/taiwan_mandarin_low_r/reports/zipvoice_4step_optimize_v1/assets/qvocoder_clean_mild/cosy_02.wav",
        "note": "同一個 ZipVoice 候選，換句子看字音和尾音穩定度。",
        "badge": "ZipVoice",
    },
    {
        "group": "Cosy / ZipVoice 對照",
        "title": "ZipVoice qint8 Vocos + mild clean：句子 3",
        "text": "我想要的不是主播腔，也不是娃娃音，是聪明、温柔、真实的声音。",
        "path": ROOT / "distillation/taiwan_mandarin_low_r/reports/zipvoice_4step_optimize_v1/assets/qvocoder_clean_mild/cosy_03.wav",
        "note": "同一個 ZipVoice 候選，較長句檢查自然度和雜訊。",
        "badge": "ZipVoice",
    },
    {
        "group": "Matcha 長訓試聽",
        "title": "Matcha 長訓：日常安撫",
        "text": "你先不要急，我们慢慢来，把事情一件一件处理好。",
        "path": ROOT / "distillation/taiwan_mandarin_low_r/reports/matcha_cosy_app_slot_v1/assets/matcha_long_1.wav",
        "note": "Matcha 15k step 長訓報告樣本，這條路的優點是快和比較適合手機，缺點是自然度還要打磨。",
        "badge": "Matcha",
    },
    {
        "group": "Matcha 長訓試聽",
        "title": "Matcha 長訓：自然回覆",
        "text": "我刚刚看了一下，应该不是你的问题，你不用太担心。",
        "path": ROOT / "distillation/taiwan_mandarin_low_r/reports/matcha_cosy_app_slot_v1/assets/matcha_long_2.wav",
        "note": "換句子檢查斷詞、語速、清晰度。",
        "badge": "Matcha",
    },
    {
        "group": "Matcha 長訓試聽",
        "title": "Matcha 長訓：溫柔陪伴",
        "text": "没关系啦，你先讲，我在这边听，真的不用紧张。",
        "path": ROOT / "distillation/taiwan_mandarin_low_r/reports/matcha_cosy_app_slot_v1/assets/matcha_long_3.wav",
        "note": "換句子檢查台灣感和句尾自然度。",
        "badge": "Matcha",
    },
]


def wav_stats(path: Path) -> dict:
    framerate, data = wavfile.read(str(path))
    channels = 1 if data.ndim == 1 else data.shape[1]
    nframes = data.shape[0]

    if np.issubdtype(data.dtype, np.floating):
        samples = data.astype(np.float64)
    elif np.issubdtype(data.dtype, np.integer):
        max_abs = float(max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max))
        samples = data.astype(np.float64) / max_abs
    else:
        samples = data.astype(np.float64)
    if samples.ndim > 1:
        samples = samples[:, 0]

    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms_db = 20 * math.log10(max(rms, 1e-12))
    peak_db = 20 * math.log10(max(peak, 1e-12))

    return {
        "size_bytes": path.stat().st_size,
        "duration": nframes / framerate if framerate else 0,
        "sample_rate": framerate,
        "channels": channels,
        "rms_db": rms_db,
        "peak_db": peak_db,
    }


def fmt_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    return f"{n / 1024:.0f} KB"


def fmt_db(v):
    return "n/a" if v is None else f"{v:.1f} dBFS"


def audio_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    total_audio_bytes = 0
    for entry in ENTRIES:
        path = Path(entry["path"])
        if not path.exists():
            missing.append(str(path))
            continue
        stats = wav_stats(path)
        total_audio_bytes += stats["size_bytes"]
        rows.append({**entry, "stats": stats, "data_url": audio_data_url(path)})

    groups = []
    for row in rows:
        if row["group"] not in groups:
            groups.append(row["group"])

    cards_html = []
    for group in groups:
        cards_html.append(f"<section><h2>{html.escape(group)}</h2><div class=\"grid\">")
        for row in [r for r in rows if r["group"] == group]:
            stats = row["stats"]
            cards_html.append(
                f"""
        <article class="card">
          <div class="topline"><span class="badge">{html.escape(row["badge"])}</span><span>{fmt_size(stats["size_bytes"])}</span></div>
          <h3>{html.escape(row["title"])}</h3>
          <p class="text">{html.escape(row["text"])}</p>
          <audio controls preload="metadata" src="{row["data_url"]}"></audio>
          <dl>
            <div><dt>音訊長度</dt><dd>{stats["duration"]:.2f}s</dd></div>
            <div><dt>取樣率</dt><dd>{stats["sample_rate"]} Hz</dd></div>
            <div><dt>RMS</dt><dd>{fmt_db(stats["rms_db"])}</dd></div>
            <div><dt>Peak</dt><dd>{fmt_db(stats["peak_db"])}</dd></div>
          </dl>
          <p class="note">{html.escape(row["note"])}</p>
        </article>
                """
            )
        cards_html.append("</div></section>")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "entries": len(rows),
        "total_audio_bytes": total_audio_bytes,
        "missing": missing,
    }

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RedBow mobile standalone audio v1</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #26231f;
      --muted: #706b63;
      --paper: #f7f4ee;
      --panel: #fffdfa;
      --line: #ded7cb;
      --red: #b51f2e;
      --soft: #ede6da;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
      line-height: 1.55;
    }}
    main {{
      width: min(1120px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 8px 0 22px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 5vw, 48px);
      line-height: 1.08;
      letter-spacing: 0;
    }}
    .lead {{
      margin: 0;
      max-width: 760px;
      color: var(--muted);
      font-size: 17px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0 0;
    }}
    .summary div {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 12px;
    }}
    .summary dt, dl dt {{
      color: var(--muted);
      font-size: 12px;
      margin: 0;
    }}
    .summary dd, dl dd {{
      margin: 2px 0 0;
      font-weight: 650;
    }}
    h2 {{
      margin: 30px 0 12px;
      font-size: 21px;
      line-height: 1.2;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .card {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
    }}
    .topline {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      align-items: center;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border: 1px solid color-mix(in srgb, var(--red), white 55%);
      color: var(--red);
      padding: 2px 7px;
      font-weight: 700;
      background: #fff6f5;
    }}
    h3 {{
      margin: 10px 0 8px;
      font-size: 18px;
      line-height: 1.25;
    }}
    .text {{
      margin: 0 0 12px;
      font-size: 16px;
    }}
    audio {{
      width: 100%;
      display: block;
      margin: 8px 0 12px;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin: 0 0 10px;
    }}
    dl div {{
      background: var(--soft);
      padding: 8px;
      min-width: 0;
    }}
    .note {{
      color: var(--muted);
      font-size: 14px;
      margin: 0;
    }}
    .explain {{
      border-top: 1px solid var(--line);
      margin-top: 30px;
      padding-top: 20px;
      color: var(--muted);
    }}
    code {{
      background: var(--soft);
      padding: 2px 5px;
    }}
    @media (max-width: 760px) {{
      main {{ width: min(100vw - 22px, 680px); padding-top: 20px; }}
      .summary, .grid, dl {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 34px; }}
      .card {{ padding: 12px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>RedBow 手機可播試聽報告</h1>
      <p class="lead">這份 HTML 已把少量關鍵 wav 直接嵌入檔案，不再引用 Mac 本機絕對路徑或旁邊的 assets 目錄。手機離開家裡 Wi-Fi 後，只要能開到這個 HTML 檔，就能播放下面的音檔。</p>
      <dl class="summary">
        <div><dt>音檔數</dt><dd>{len(rows)}</dd></div>
        <div><dt>內嵌音檔總量</dt><dd>{fmt_size(total_audio_bytes)}</dd></div>
        <div><dt>產生時間</dt><dd>{html.escape(payload["created_at"])}</dd></div>
      </dl>
    </header>

    {"".join(cards_html)}

    <section class="explain">
      <h2>為什麼雲端版可能播不了</h2>
      <p>之前很多報告的 audio 是相對路徑，例如 <code>assets/matcha_long_1.wav</code>，或本機路徑。這在 Mac 上可用，但手機到公司後沒有那個資料夾，所以播放器會失敗。這份改成 <code>data:audio/wav;base64</code>，音檔跟 HTML 綁在一起。</p>
      <p>如果 Google Drive App 預覽頁不播放，請用「在瀏覽器開啟」或下載後用 Safari/Chrome 開。Drive 預覽有時不是完整網頁環境，但檔案本身已經是離線可播格式。</p>
      <p>目前 app 先放 Matcha-only，是因為 ZipVoice 4-step 雖然有幾個 qint8 Vocos + mild clean 樣本不錯，但實機/不同句子的電子雜訊還不穩。Matcha 的方向是速度和手機整合優先，接下來再調語速、vocoder 和後處理讓它更自然。</p>
    </section>
  </main>
  <script type="application/json" id="report-metadata">{html.escape(json.dumps(payload, ensure_ascii=False))}</script>
</body>
</html>
"""

    OUT_HTML.write_text(html_text, encoding="utf-8")
    print(OUT_HTML)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
