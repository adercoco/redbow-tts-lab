#!/usr/bin/env python3
"""Build a standalone HTML report for the CosyVoice -> ZipVoice distillation run."""

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
    / "cosy_zipvoice_true_distill_method_v1"
)
OUT_HTML = REPORT_DIR / "cosy_zipvoice_true_distill_method_v1_standalone.html"

TEACHER_DIR = (
    ROOT / "distillation" / "taiwan_mandarin_low_r" / "teacher_cosy_raw_best2_distill_v1"
)
ZIP_EGS = ROOT / "external" / "ZipVoice" / "egs" / "zipvoice"

TEST_LINES = [
    (
        "cosy_01",
        "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
        TEACHER_DIR / "audio" / "distill_0001.wav",
    ),
    (
        "cosy_02",
        "这样听起来有比较像台湾人吗？我希望尾音自然一点，不要太卷舌。",
        TEACHER_DIR / "audio" / "distill_0002.wav",
    ),
    (
        "cosy_03",
        "我想要的不是主播腔，也不是娃娃音，是聪明、温柔、真实的声音。",
        TEACHER_DIR / "audio" / "distill_0003.wav",
    ),
]

AUDIO_VARIANTS = [
    (
        "Cosy teacher",
        "CosyVoice2-0.5B 生成的 teacher target，ZipVoice 學習目標。",
        None,
    ),
    (
        "ZipVoice original",
        "原始 ZipVoice PyTorch 16-step，同 reference、同文字。",
        ZIP_EGS / "results" / "_measure_cosy_zipvoice_original_py_step16",
    ),
    (
        "True-distill PyTorch",
        "這次 Cosy corpus fine-tune 後的 checkpoint-60 model-only，16-step。",
        ZIP_EGS / "results" / "cosy_true_distill_model_only_py_step16",
    ),
    (
        "True-distill ONNX int8 8-step",
        "手機候選速度版，權重已 export ONNX/int8，steps 降到 8。",
        ZIP_EGS / "results" / "cosy_true_distill_onnx_int8_step8",
    ),
    (
        "True-distill ONNX int8 4-step",
        "最快候選；速度漂亮，但音質與字準要人工確認。",
        ZIP_EGS / "results" / "cosy_true_distill_onnx_int8_step4",
    ),
]

METRICS = [
    {
        "name": "CosyVoice2 teacher",
        "role": "老師模型，產訓練語料",
        "size": "4.5GB",
        "runtime": "未做手機部署",
        "peak": "未量測",
        "time": "產 corpus 用，不作手機推論基準",
        "note": "用來生 64 句 teacher audio；不是手機端模型。",
    },
    {
        "name": "ZipVoice original PyTorch",
        "role": "原始學生 baseline",
        "size": "468MB",
        "runtime": "PyTorch CPU 16-step",
        "peak": "1.71GB peak RSS",
        "time": "69.98s / 3 句，平均 RTF 4.59",
        "note": "同 reference、同三句，作為聲音與速度 baseline。",
    },
    {
        "name": "True-distill PyTorch",
        "role": "真正 fine-tune 後學生",
        "size": "468MB model-only；訓練 checkpoint 1.8GB",
        "runtime": "PyTorch CPU 16-step",
        "peak": "1.67GB peak RSS",
        "time": "69.89s / 3 句，平均 RTF 4.58",
        "note": "權重已被 Cosy teacher corpus 更新；速度不會因 fine-tune 自動變快。",
    },
    {
        "name": "True-distill ONNX int8 16-step",
        "role": "部署候選，高品質 steps",
        "size": "175.8MB int8 + vocoder",
        "runtime": "ONNX Runtime CPU",
        "peak": "1.55GB peak RSS",
        "time": "27.61s / 3 句，平均 RTF 1.68",
        "note": "比 PyTorch 快很多，但還不是即時。",
    },
    {
        "name": "True-distill ONNX int8 8-step",
        "role": "手機候選平衡版",
        "size": "175.8MB int8 + vocoder",
        "runtime": "ONNX Runtime CPU",
        "peak": "1.54GB peak RSS",
        "time": "15.44s / 3 句，平均 RTF 0.86",
        "note": "目前比較有機會成為手機 demo 預設。",
    },
    {
        "name": "True-distill ONNX int8 4-step",
        "role": "最快候選版",
        "size": "175.8MB int8 + vocoder",
        "runtime": "ONNX Runtime CPU",
        "peak": "1.55GB peak RSS",
        "time": "9.78s / 3 句，平均 RTF 0.45",
        "note": "速度好；音質、字準與尾音需要人工聽感確認。",
    },
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def audio_src(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{data}"


def audio_control(path: Path) -> str:
    if not path.exists():
        return '<div class="missing">missing audio</div>'
    return f'<audio controls preload="none" src="{audio_src(path)}"></audio>'


def metric_cards() -> str:
    cards = []
    for item in METRICS:
        cards.append(
            f"""
            <article class="metric-card">
              <div class="metric-title">{html.escape(item["name"])}</div>
              <div class="metric-role">{html.escape(item["role"])}</div>
              <dl>
                <div><dt>模型大小</dt><dd>{html.escape(item["size"])}</dd></div>
                <div><dt>執行方式</dt><dd>{html.escape(item["runtime"])}</dd></div>
                <div><dt>峰值記憶體</dt><dd>{html.escape(item["peak"])}</dd></div>
                <div><dt>生成時間</dt><dd>{html.escape(item["time"])}</dd></div>
              </dl>
              <p>{html.escape(item["note"])}</p>
            </article>
            """
        )
    return "\n".join(cards)


def audio_rows() -> str:
    rows = []
    for audio_id, text, teacher_audio in TEST_LINES:
        cells = []
        for label, desc, directory in AUDIO_VARIANTS:
            wav = teacher_audio if directory is None else directory / f"{audio_id}.wav.wav"
            cells.append(
                f"""
                <div class="audio-cell">
                  <div class="audio-label">{html.escape(label)}</div>
                  <div class="audio-desc">{html.escape(desc)}</div>
                  {audio_control(wav)}
                </div>
                """
            )
        rows.append(
            f"""
            <section class="audio-row">
              <div class="line-text">{html.escape(text)}</div>
              <div class="audio-grid">{''.join(cells)}</div>
            </section>
            """
        )
    return "\n".join(rows)


def build_html() -> str:
    generated = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CosyVoice → ZipVoice 真蒸餾做法報告</title>
  <style>
    :root {{
      --bg: #f7f4ef;
      --paper: #fffdf8;
      --ink: #201b17;
      --muted: #6d6258;
      --line: #ded5c8;
      --accent: #a33b28;
      --accent-2: #255e63;
      --soft: #efe7dc;
      --good: #2f6f4e;
      --warn: #9a5b1f;
      --radius: 8px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", "PingFang TC", sans-serif;
      line-height: 1.62;
      letter-spacing: 0;
    }}
    main {{
      width: min(1120px, 100%);
      margin: 0 auto;
      padding: 24px 16px 56px;
    }}
    header {{
      padding: 26px 0 18px;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 8px 0 10px;
      font-size: clamp(30px, 6vw, 56px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    .subtitle {{
      max-width: 820px;
      margin: 0;
      color: var(--muted);
      font-size: 17px;
    }}
    section.band {{
      padding: 26px 0;
      border-bottom: 1px solid var(--line);
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 24px;
      line-height: 1.22;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 18px;
      line-height: 1.3;
    }}
    p {{ margin: 0 0 12px; }}
    .status {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .pill {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 12px;
    }}
    .pill strong {{
      display: block;
      font-size: 15px;
      color: var(--accent-2);
    }}
    .pill span {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 14px;
    }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .step {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 16px;
    }}
    .step-num {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 800;
    }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #211d19;
      color: #f8efe4;
      border-radius: var(--radius);
      padding: 14px;
      font-size: 12px;
      line-height: 1.5;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 14px;
    }}
    .metric-title {{
      font-size: 18px;
      font-weight: 800;
    }}
    .metric-role {{
      color: var(--accent-2);
      font-size: 13px;
      font-weight: 700;
      margin: 2px 0 10px;
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
      font-weight: 700;
    }}
    dd {{
      margin: 1px 0 0;
      font-size: 15px;
      font-weight: 650;
    }}
    .metric-card p {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
      display: table;
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
      font-size: 12px;
      color: var(--muted);
    }}
    tr:last-child td {{ border-bottom: none; }}
    .audio-row {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 14px;
      margin-bottom: 12px;
    }}
    .line-text {{
      font-size: 17px;
      font-weight: 750;
      margin-bottom: 12px;
    }}
    .audio-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }}
    .audio-cell {{
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 10px;
      background: #fffaf1;
    }}
    .audio-label {{
      font-size: 14px;
      font-weight: 800;
    }}
    .audio-desc {{
      color: var(--muted);
      font-size: 12px;
      min-height: 58px;
    }}
    audio {{
      width: 100%;
      margin-top: 8px;
    }}
    .callout {{
      border-left: 4px solid var(--accent);
      background: #fff8ed;
      padding: 14px;
      border-radius: 0 var(--radius) var(--radius) 0;
    }}
    .good {{ color: var(--good); font-weight: 800; }}
    .warn {{ color: var(--warn); font-weight: 800; }}
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
      overflow-wrap: anywhere;
      font-size: 13px;
    }}
    @media (max-width: 860px) {{
      main {{ padding: 18px 12px 42px; }}
      .status, .steps, .metrics, .paths {{ grid-template-columns: 1fr; }}
      .audio-grid {{ grid-template-columns: 1fr; }}
      .audio-desc {{ min-height: 0; }}
      table, thead, tbody, th, td, tr {{ display: block; }}
      thead {{ display: none; }}
      tr {{
        border-bottom: 1px solid var(--line);
        padding: 8px 0;
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
    <div class="eyebrow">Standalone report · generated {html.escape(generated)} Asia/Taipei</div>
    <h1>CosyVoice → ZipVoice 真蒸餾做法報告</h1>
    <p class="subtitle">這份報告只講這次真正有更新 ZipVoice 權重的流程：用 CosyVoice2 生成 teacher corpus，將 ZipVoice fine-tune 成學生模型，再輸出 model-only 與 ONNX int8 版本做手機候選測試。</p>
    <div class="status">
      <div class="pill"><strong>已完成</strong><span>ZipVoice 權重真 fine-tune，loss 從 0.02234 降到 0.02037。</span></div>
      <div class="pill"><strong>已部署化</strong><span>checkpoint-60 已抽成 model-only，並 export ONNX/int8。</span></div>
      <div class="pill"><strong>仍有限制</strong><span>官方 ZipVoice 推論仍需要 prompt audio，不是純文字固定聲線。</span></div>
    </div>
  </header>

  <section class="band">
    <h2>一句話結論</h2>
    <div class="callout">
      <p><span class="good">這次是真蒸餾第一版：</span>ZipVoice 真的用 CosyVoice teacher 語音訓練過，不是只拿 reference zero-shot 換聲音。</p>
      <p><span class="warn">但它不是終局版：</span>它還不是更小架構學生，也還不能完全不給 reference。手機候選目前應看 ONNX int8 8-step 或 4-step。</p>
    </div>
  </section>

  <section class="band">
    <h2>蒸餾定義</h2>
    <p>這次做的是 <strong>teacher-corpus supervised distillation</strong>：老師 CosyVoice2 先生成目標聲音語料，學生 ZipVoice 用這批語音重新 fine-tune。學生的權重會改變，所以它是真訓練。</p>
    <p>它還不是 <strong>architecture distillation</strong> 或 <strong>no-reference single-speaker TTS</strong>。如果要讓模型更小、更快、完全不需要 prompt audio，下一階段要訓練固定聲線 conditioning 或更小 ZipVoice/ZipVoiceDistill 架構。</p>
  </section>

  <section class="band">
    <h2>完整做法</h2>
    <div class="steps">
      <article class="step"><div class="step-num">STEP 1</div><h3>整理可用女聲 reference</h3><p>使用先前從下載資料夾整理出的女聲片段，排除男聲、配樂、唱歌與重疊聲音。這次 teacher generation 使用 <code>raw_best2_7s.wav</code> 作為 CosyVoice prompt reference。</p></article>
      <article class="step"><div class="step-num">STEP 2</div><h3>CosyVoice2 生成 teacher corpus</h3><p>用 <code>CosyVoice2-0.5B</code> 對 <code>distill_texts_v1.jsonl</code> 生成 64 句台灣女生風格語音。輸出在 <code>{html.escape(rel(TEACHER_DIR))}</code>。</p></article>
      <article class="step"><div class="step-num">STEP 3</div><h3>轉成 ZipVoice 訓練資料</h3><p>把 teacher corpus 切成 56 句 train / 8 句 dev，產 TSV，接著用 ZipVoice/Lhotse 產 manifest、tokens、fbank。</p></article>
      <article class="step"><div class="step-num">STEP 4</div><h3>從原始 ZipVoice fine-tune</h3><p>從官方 <code>download/zipvoice/model.pt</code> 載入 122.7M parameters，用 Cosy teacher fbank 做 60 iterations 短訓練，每 20 iterations 存 checkpoint。</p></article>
      <article class="step"><div class="step-num">STEP 5</div><h3>選 checkpoint-60</h3><p>validation loss 在 60 iterations 最低，選 <code>checkpoint-60.pt</code>。訓練 checkpoint 1.8GB，因為包含 optimizer、scheduler、model_avg。</p></article>
      <article class="step"><div class="step-num">STEP 6</div><h3>抽 model-only</h3><p>部署不需要 optimizer 等訓練狀態，所以抽出 <code>{{"model": state_dict}}</code> 成 <code>checkpoint-60-model-only.pt</code>，大小回到 468MB。</p></article>
      <article class="step"><div class="step-num">STEP 7</div><h3>Export ONNX/int8</h3><p>用 ZipVoice 官方 <code>onnx_export.py</code> 轉出 <code>text_encoder_int8.onnx</code> 與 <code>fm_decoder_int8.onnx</code>。加上 vocoder 後部署核心約 175.8MB。</p></article>
      <article class="step"><div class="step-num">STEP 8</div><h3>測 16 / 8 / 4 steps</h3><p>同三句、同 reference 測 PyTorch 與 ONNX int8。steps 越少越快，但聲音穩定度通常下降。</p></article>
    </div>
  </section>

  <section class="band">
    <h2>核心訓練命令</h2>
    <pre>PYTHONPATH=../../:$PYTHONPATH /Users/ader/Documents/App/.venv-zipvoice/bin/python -m zipvoice.bin.train_zipvoice \\
  --world-size 1 \\
  --use-fp16 0 \\
  --finetune 1 \\
  --base-lr 0.00005 \\
  --num-iters 60 \\
  --save-every-n 20 \\
  --max-duration 80 \\
  --max-len 15 \\
  --num-buckets 4 \\
  --num-workers 0 \\
  --model-config download/zipvoice/model.json \\
  --checkpoint download/zipvoice/model.pt \\
  --tokenizer emilia \\
  --lang default \\
  --token-file download/zipvoice/tokens.txt \\
  --dataset custom \\
  --train-manifest data/fbank/custom-cosy_cuts_train.jsonl.gz \\
  --dev-manifest data/fbank/custom-cosy_cuts_dev.jsonl.gz \\
  --exp-dir exp/zipvoice_cosy_teacher_smoke</pre>
  </section>

  <section class="band">
    <h2>訓練結果</h2>
    <table>
      <thead><tr><th>階段</th><th>global batch</th><th>validation loss</th><th>判讀</th></tr></thead>
      <tbody>
        <tr><td data-label="階段">初始</td><td data-label="global batch">0</td><td data-label="validation loss">0.02234</td><td data-label="判讀">原始 ZipVoice 載入後，尚未學 Cosy corpus。</td></tr>
        <tr><td data-label="階段">checkpoint-20</td><td data-label="global batch">20</td><td data-label="validation loss">0.02427</td><td data-label="判讀">短期波動，未選。</td></tr>
        <tr><td data-label="階段">checkpoint-40</td><td data-label="global batch">40</td><td data-label="validation loss">0.02115</td><td data-label="判讀">開始明顯學到 teacher corpus。</td></tr>
        <tr><td data-label="階段">checkpoint-60</td><td data-label="global batch">60</td><td data-label="validation loss">0.02037</td><td data-label="判讀">本輪最佳，作為 true-distill v1。</td></tr>
      </tbody>
    </table>
  </section>

  <section class="band">
    <h2>模型大小、記憶體、速度</h2>
    <div class="metrics">{metric_cards()}</div>
    <p style="margin-top:14px;color:var(--muted)">補充：這裡的生成時間是同三句測試總時間與平均 RTF。peak RSS 是 macOS <code>/usr/bin/time -l</code> 測到的 Python/ONNX 測試峰值，不等於 iPhone/Android 原生 runtime 的最終數字，但可用來比較方向。</p>
  </section>

  <section class="band">
    <h2>試聽比較</h2>
    <p>每一列是同一句台詞。建議先聽 Cosy teacher，再聽 original ZipVoice，最後聽 true-distill 的 PyTorch / ONNX 8-step / ONNX 4-step。</p>
    {audio_rows()}
  </section>

  <section class="band">
    <h2>為什麼模型變小，記憶體沒有大降</h2>
    <p>ONNX int8 讓權重檔從 468MB 級別降到約 175.8MB，但 peak RSS 仍在 1.5GB 左右，原因是 runtime 不是只載權重。它還包括 ONNX Runtime arena、decoder 中間 activation、fbank/prompt features、vocoder、thread workspace，以及 Python 測試程式本身。</p>
    <p>真正要讓手機峰值更低，需要原生 sherpa-onnx/mobile runtime 測試、降低 max sequence、切句生成、vocoder 最佳化，或訓練更小的 ZipVoice 架構學生。</p>
  </section>

  <section class="band">
    <h2>目前限制與下一步</h2>
    <div class="callout">
      <p><strong>限制 1：</strong>ZipVoice 官方推論仍要求 <code>prompt-wav + prompt-text</code>。這次是權重真蒸餾，但還不是完全 no-reference。</p>
      <p><strong>限制 2：</strong>64 句 teacher corpus 只夠做 smoke / v1。若要聲線穩、台灣腔更明顯，建議 teacher corpus 至少拉到數百句，並加入更多同聲線情緒與句長。</p>
      <p><strong>限制 3：</strong>fine-tune 會改聲音，但不會自動加速。加速主要靠 ONNX/int8、few-step、或訓練 ZipVoiceDistill 小步數模型。</p>
    </div>
    <p>建議下一步：以這次 Cosy teacher corpus 擴到 300-1000 句，訓練更長；同時訓練 ZipVoiceDistill/few-step 版本，目標直接讓 4-step 或 6-step 有 8-step 的音質。</p>
  </section>

  <section class="band">
    <h2>重要檔案</h2>
    <div class="paths">
      <div class="path"><strong>Teacher corpus</strong><br><code>{html.escape(rel(TEACHER_DIR))}</code></div>
      <div class="path"><strong>ZipVoice train exp</strong><br><code>{html.escape(rel(ZIP_EGS / "exp" / "zipvoice_cosy_teacher_smoke"))}</code></div>
      <div class="path"><strong>Model-only checkpoint</strong><br><code>{html.escape(rel(ZIP_EGS / "exp" / "zipvoice_cosy_teacher_smoke" / "checkpoint-60-model-only.pt"))}</code></div>
      <div class="path"><strong>ONNX/int8 export</strong><br><code>{html.escape(rel(ZIP_EGS / "exp" / "zipvoice_cosy_teacher_smoke_onnx_ckpt60"))}</code></div>
      <div class="path"><strong>Test TSV</strong><br><code>{html.escape(rel(ZIP_EGS / "test_cosy_teacher_smoke.tsv"))}</code></div>
      <div class="path"><strong>Report builder</strong><br><code>{html.escape(rel(ROOT / "tools" / "build_cosy_true_distill_method_report.py"))}</code></div>
    </div>
  </section>
</main>
</body>
</html>
"""


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(), encoding="utf-8")
    print(OUT_HTML)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
