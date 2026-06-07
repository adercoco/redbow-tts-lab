#!/usr/bin/env python3
"""Build a standalone Cosy teacher vs ZipVoice true-distill 4-step report."""

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
    / "cosy_vs_zipvoice_true_distill_4step_v1"
)
OUT_HTML = REPORT_DIR / "cosy_vs_zipvoice_true_distill_4step_v1_standalone.html"

TEACHER_DIR = (
    ROOT / "distillation" / "taiwan_mandarin_low_r" / "teacher_cosy_raw_best2_distill_v1"
)
ZIP_EGS = ROOT / "external" / "ZipVoice" / "egs" / "zipvoice"
ZIP_4STEP_DIR = ZIP_EGS / "results" / "cosy_true_distill_onnx_int8_step4"

LINES = [
    (
        "cosy_01",
        "distill_0001",
        "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
        "9.44s Cosy run / 2.48s ZipVoice 4-step",
    ),
    (
        "cosy_02",
        "distill_0002",
        "这样听起来有比较像台湾人吗？我希望尾音自然一点，不要太卷舌。",
        "10.74s Cosy run / 2.25s ZipVoice 4-step",
    ),
    (
        "cosy_03",
        "distill_0003",
        "我想要的不是主播腔，也不是娃娃音，是聪明、温柔、真实的声音。",
        "10.59s Cosy run / 2.23s ZipVoice 4-step",
    ),
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def audio_src(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def audio(path: Path) -> str:
    if not path.exists():
        return '<div class="missing">missing audio</div>'
    return f'<audio controls preload="none" src="{audio_src(path)}"></audio>'


def rows() -> str:
    out = []
    for zip_id, teacher_id, text, timing in LINES:
        teacher = TEACHER_DIR / "audio" / f"{teacher_id}.wav"
        student = ZIP_4STEP_DIR / f"{zip_id}.wav.wav"
        out.append(
            f"""
            <section class="listen-row">
              <div class="line">
                <div class="tag">測試句</div>
                <p>{html.escape(text)}</p>
                <span>{html.escape(timing)}</span>
              </div>
              <div class="compare">
                <article>
                  <h3>CosyVoice2 teacher</h3>
                  <p>老師目標聲音。自然度與台灣女生感目前以這個為上限。</p>
                  {audio(teacher)}
                </article>
                <article>
                  <h3>ZipVoice true-distill ONNX int8 4-step</h3>
                  <p>這次真正 fine-tune 後的學生模型，手機速度候選版。</p>
                  {audio(student)}
                </article>
              </div>
            </section>
            """
        )
    return "\n".join(out)


def build() -> str:
    generated = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cosy vs ZipVoice 4-step 真蒸餾對比</title>
  <style>
    :root {{
      --bg: #f6f2eb;
      --paper: #fffdf8;
      --ink: #211c18;
      --muted: #71675e;
      --line: #ddd2c4;
      --accent: #ad3d2b;
      --teal: #285c64;
      --soft: #efe6d9;
      --good: #2d6f4f;
      --warn: #986019;
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
      width: min(1080px, 100%);
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
      line-height: 1.05;
      letter-spacing: 0;
    }}
    .subtitle {{
      max-width: 780px;
      color: var(--muted);
      font-size: 17px;
      margin: 0;
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
      font-size: 17px;
      line-height: 1.3;
    }}
    p {{ margin: 0 0 10px; }}
    .hero-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .stat {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
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
    .verdict {{
      background: #fff8ed;
      border-left: 4px solid var(--accent);
      border-radius: 0 var(--radius) var(--radius) 0;
      padding: 14px;
    }}
    .two {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 14px;
    }}
    .card .role {{
      color: var(--teal);
      font-size: 13px;
      font-weight: 800;
      margin-bottom: 10px;
    }}
    dl {{
      margin: 0;
      display: grid;
      gap: 8px;
    }}
    dl div {{
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }}
    dt {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}
    dd {{
      margin: 1px 0 0;
      font-size: 15px;
      font-weight: 650;
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
    .listen-row {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 14px;
      margin-bottom: 12px;
    }}
    .line .tag {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
    }}
    .line p {{
      font-size: 17px;
      font-weight: 750;
      margin: 2px 0 2px;
    }}
    .line span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .compare {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .compare article {{
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fffaf1;
      padding: 10px;
    }}
    .compare p {{
      min-height: 44px;
      color: var(--muted);
      font-size: 13px;
    }}
    audio {{
      width: 100%;
      margin-top: 8px;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
    }}
    .paths {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .path {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 10px;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .good {{ color: var(--good); font-weight: 800; }}
    .warn {{ color: var(--warn); font-weight: 800; }}
    @media (max-width: 820px) {{
      main {{ padding: 18px 12px 44px; }}
      .hero-grid, .two, .compare, .paths {{ grid-template-columns: 1fr; }}
      .compare p {{ min-height: 0; }}
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
    <h1>Cosy vs ZipVoice 4-step 真蒸餾對比</h1>
    <p class="subtitle">這份是專門給 4-step 看的結果報告：左邊是 CosyVoice2 老師聲音，右邊是這次真正 fine-tune 後的 ZipVoice ONNX int8 4-step 學生模型。</p>
    <div class="hero-grid">
      <div class="stat"><span>結果狀態</span><strong>有結果</strong></div>
      <div class="stat"><span>學生模型大小</span><strong>175.8MB</strong></div>
      <div class="stat"><span>4-step 生成</span><strong>9.78s / 3句</strong></div>
      <div class="stat"><span>4-step peak RSS</span><strong>1.55GB</strong></div>
    </div>
  </header>

  <section class="band">
    <h2>結論</h2>
    <div class="verdict">
      <p><span class="good">新的真蒸餾已經有可聽結果：</span>ZipVoice checkpoint-60 已 export 成 ONNX int8，4-step 可以在這台 Mac CPU 上用約 0.45 RTF 生成，速度已經比 Cosy teacher 快很多。</p>
      <p><span class="warn">但 4-step 是速度優先版：</span>它是否保留足夠 Cosy 的溫柔、清亮、台灣低卷舌感，要以這份試聽為準。若 4-step 音質不夠，8-step 是目前比較穩的備選。</p>
    </div>
  </section>

  <section class="band">
    <h2>關鍵資訊</h2>
    <div class="two">
      <article class="card">
        <h3>CosyVoice2 teacher</h3>
        <div class="role">老師模型 · 聲音目標</div>
        <dl>
          <div><dt>模型大小</dt><dd>4.5GB</dd></div>
          <div><dt>推論記憶體</dt><dd>5.73GB max RSS；6.47GB peak memory footprint</dd></div>
          <div><dt>生成時間</dt><dd>43.00s / 3句；單句 9.44s、10.74s、10.59s</dd></div>
          <div><dt>用途</dt><dd>產 teacher corpus，不適合直接塞手機離線 demo。</dd></div>
        </dl>
      </article>
      <article class="card">
        <h3>ZipVoice true-distill ONNX int8 4-step</h3>
        <div class="role">學生模型 · 手機速度候選</div>
        <dl>
          <div><dt>部署核心大小</dt><dd>175.8MB = text encoder 5.3MB + decoder 118.8MB + vocoder 51.6MB</dd></div>
          <div><dt>完整 export 目錄</dt><dd>596MB；包含 fp32 ONNX 與 int8 ONNX，手機可只包 int8 核心。</dd></div>
          <div><dt>推論記憶體</dt><dd>1.55GB max RSS；1.47GB peak memory footprint</dd></div>
          <div><dt>生成時間</dt><dd>9.78s / 3句；平均 RTF 0.45</dd></div>
        </dl>
      </article>
    </div>
  </section>

  <section class="band">
    <h2>速度與大小對比</h2>
    <table>
      <thead><tr><th>項目</th><th>Cosy teacher</th><th>ZipVoice 4-step 真蒸餾</th><th>判讀</th></tr></thead>
      <tbody>
        <tr><td data-label="項目">模型大小</td><td data-label="Cosy teacher">4.5GB</td><td data-label="ZipVoice 4-step 真蒸餾">175.8MB deploy core</td><td data-label="判讀">學生約為老師的 3.8%。</td></tr>
        <tr><td data-label="項目">Peak RSS</td><td data-label="Cosy teacher">5.73GB</td><td data-label="ZipVoice 4-step 真蒸餾">1.55GB</td><td data-label="判讀">學生約為老師的 27%。</td></tr>
        <tr><td data-label="項目">三句總生成</td><td data-label="Cosy teacher">43.00s</td><td data-label="ZipVoice 4-step 真蒸餾">9.78s</td><td data-label="判讀">學生約 4.4x 快。</td></tr>
        <tr><td data-label="項目">平均 RTF</td><td data-label="Cosy teacher">約 1.79</td><td data-label="ZipVoice 4-step 真蒸餾">0.45</td><td data-label="判讀">4-step 已低於 1.0 RTF，有即時化潛力。</td></tr>
        <tr><td data-label="項目">聲音上限</td><td data-label="Cosy teacher">目前目標聲音</td><td data-label="ZipVoice 4-step 真蒸餾">需聽感確認</td><td data-label="判讀">速度換音質，4-step 要看是否掉字或變粗糙。</td></tr>
      </tbody>
    </table>
  </section>

  <section class="band">
    <h2>試聽：Cosy vs 4-step</h2>
    <p>每一句都先聽 Cosy teacher，再聽 ZipVoice 4-step。重點聽：尾音、卷舌、女聲清亮感、字有沒有糊掉。</p>
    {rows()}
  </section>

  <section class="band">
    <h2>這版怎麼蒸餾出來的</h2>
    <p>流程是：用 CosyVoice2-0.5B 生成 64 句 teacher corpus，切成 56 train / 8 dev；從官方 ZipVoice 468MB checkpoint 起跑，對這批 Cosy teacher audio 做 60 iterations fine-tune；validation loss 從 <code>0.02234</code> 降到 <code>0.02037</code>，選 <code>checkpoint-60</code>。</p>
    <p>接著把 1.8GB 訓練 checkpoint 抽成 <code>checkpoint-60-model-only.pt</code>，大小回到 468MB，再 export 成 ONNX/int8。手機包裝時不需要 fp32 ONNX，所以核心可以抓 int8 encoder、int8 decoder、vocoder，大約 175.8MB。</p>
  </section>

  <section class="band">
    <h2>目前判斷</h2>
    <div class="verdict">
      <p><strong>可以繼續走 4-step，但不能只看速度。</strong> 如果你聽起來 4-step 還有 Cosy 的女聲質感，那它就是目前最有價值的手機候選；如果 4-step 太粗、尾音掉、字怪，下一個應該用 8-step 當預設，再訓練 few-step distill 讓 4-step 追上 8-step。</p>
    </div>
  </section>

  <section class="band">
    <h2>檔案位置</h2>
    <div class="paths">
      <div class="path"><strong>本報告</strong><br><code>{html.escape(rel(OUT_HTML))}</code></div>
      <div class="path"><strong>Cosy teacher corpus</strong><br><code>{html.escape(rel(TEACHER_DIR))}</code></div>
      <div class="path"><strong>4-step outputs</strong><br><code>{html.escape(rel(ZIP_4STEP_DIR))}</code></div>
      <div class="path"><strong>ONNX/int8 model</strong><br><code>{html.escape(rel(ZIP_EGS / "exp" / "zipvoice_cosy_teacher_smoke_onnx_ckpt60"))}</code></div>
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
