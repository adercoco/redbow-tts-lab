#!/usr/bin/env python3
"""Build a phone-readable ZipVoice distillation listening report."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
OUT = BASE / "reports" / "zipvoice_three_way"
AUDIO = OUT / "audio" / "zipvoice_distill_v1"
ZIP_EGS = ROOT / "external" / "ZipVoice" / "egs" / "zipvoice"
TEACHER_COMPARE = OUT / "teacher_qwen3_1p7b_compare" / "audio"

TEST_LINES = [
    {
        "id": "smoke_01",
        "zip_file": "smoke_01.wav.wav",
        "teacher_file": "smoke_01.wav",
        "text": "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
    },
    {
        "id": "smoke_02",
        "zip_file": "smoke_02.wav.wav",
        "teacher_file": "smoke_02.wav",
        "text": "我知道你现在有点紧张，可是先不要急着证明自己，慢慢说，我会听完。",
    },
    {
        "id": "smoke_03",
        "zip_file": "smoke_03.wav.wav",
        "teacher_file": "smoke_03.wav",
        "text": "这件事我们先放在同一个地方整理，等线索够清楚，再决定下一步要怎么做。",
    },
]

VARIANTS = [
    {
        "id": "teacher_qwen3_1p7b",
        "name": "1.7B 老師聲音",
        "tag": "Qwen3 VoiceDesign",
        "src": TEACHER_COMPARE,
        "file_key": "teacher_file",
        "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
        "size": "約 2.2GB cache；GB 級手機包，不適合一般離線 app",
        "memory": "2.33GB max RSS；這次同三句 MLX 測試",
        "one_line": "約 1.7-2.1s / 句；模型已載入後",
        "time": "冷啟動 command 7.28s / 3 句；含模型載入",
        "rtf": "老師基準，不看 RTF 排名",
        "quality": "目前最接近你選定的台灣國語低卷舌女聲，是所有學生模型要追的聲音。",
    },
    {
        "id": "zipvoice_int8_step8",
        "name": "ZipVoice-Distill ONNX int8",
        "tag": "目前手機候選",
        "src": ZIP_EGS / "results" / "qwen_teacher_distill_onnx_int8_ckpt10_step8",
        "file_key": "zip_file",
        "model": "ZipVoice-Distill checkpoint-10，ONNX dynamic int8",
        "size": "text encoder 5.3MB + decoder 119MB + vocoder 52MB = 約 177MB",
        "memory": "1.22GB max RSS；Python ONNX CPU 測試",
        "one_line": "約 2.7s / 句；模型已載入後",
        "time": "冷啟動 command 11.21s / 3 句；核心 RTF 0.403-0.414；8 steps",
        "rtf": "比較穩的 int8 設定",
        "quality": "比 4-step 穩，掉字風險較低；如果聽起來還不像，主要要靠更長訓練與挑 checkpoint。",
    },
    {
        "id": "zipvoice_int8_step6",
        "name": "更快優化版",
        "tag": "同模型 6-step",
        "src": ZIP_EGS / "results" / "qwen_teacher_distill_onnx_int8_ckpt10_step6",
        "file_key": "zip_file",
        "model": "同一個 ZipVoice-Distill ONNX int8，推論 steps 從 8 降到 6",
        "size": "同上，約 177MB；這是速度優化，不是新小模型",
        "memory": "1.21GB max RSS；Python ONNX CPU 測試",
        "one_line": "約 2.1s / 句；模型已載入後",
        "time": "冷啟動 command 9.10s / 3 句；核心 RTF 0.311；6 steps",
        "rtf": "比 8-step 快約 25%",
        "quality": "有機會差不多，但咬字穩定性可能略降；這欄是下一個最值得聽的速度/品質折衷。",
    },
]

LOW_STEP_VARIANTS = [
    {
        "id": "stage1_step2",
        "name": "Stage1 2-step",
        "tag": "原蒸餾",
        "src": ZIP_EGS / "results" / "qwen_teacher_stage1_ckpt10_onnx_int8_step2_th4",
        "file_key": "zip_file",
        "rtf": "0.1070",
        "memory": "1.22GB max RSS",
        "note": "最快，但最需要檢查漏字和發音糊掉。",
    },
    {
        "id": "stage1_step3",
        "name": "Stage1 3-step",
        "tag": "原蒸餾",
        "src": ZIP_EGS / "results" / "qwen_teacher_stage1_ckpt10_onnx_int8_step3_th4",
        "file_key": "zip_file",
        "rtf": "0.1523",
        "memory": "1.22GB max RSS",
        "note": "速度很好，聽感若穩就有機會當手機預設。",
    },
    {
        "id": "stage1_step4",
        "name": "Stage1 4-step",
        "tag": "原蒸餾",
        "src": ZIP_EGS / "results" / "_speed_matrix_int8_step4_th4",
        "file_key": "zip_file",
        "rtf": "0.2125",
        "memory": "約 1.22GB max RSS",
        "note": "比 6-step 快很多，是目前保守加速候選。",
    },
    {
        "id": "stage2_step2",
        "name": "Stage2 2-step",
        "tag": "few-step 新版",
        "src": ZIP_EGS / "results" / "qwen_teacher_stage2_fewstep10_onnx_int8_step2_th4",
        "file_key": "zip_file",
        "rtf": "0.1066",
        "memory": "1.22GB max RSS",
        "note": "stage2 早期 checkpoint；速度跟 stage1 幾乎一樣，重點聽穩定性。",
    },
    {
        "id": "stage2_step3",
        "name": "Stage2 3-step",
        "tag": "few-step 新版",
        "src": ZIP_EGS / "results" / "qwen_teacher_stage2_fewstep10_onnx_int8_step3_th4",
        "file_key": "zip_file",
        "rtf": "0.1505",
        "memory": "1.22GB max RSS",
        "note": "目前最值得聽的激進候選：速度快，仍可能保留可懂度。",
    },
    {
        "id": "stage2_step4",
        "name": "Stage2 4-step",
        "tag": "few-step 新版",
        "src": ZIP_EGS / "results" / "qwen_teacher_stage2_fewstep10_onnx_int8_step4_th4",
        "file_key": "zip_file",
        "rtf": "0.1990",
        "memory": "1.22GB max RSS",
        "note": "比原 4-step 稍快；如果字穩，這會比 6-step 更適合 demo app。",
    },
]

LOSSES = [
    ("ZipVoice fine-tune start", "0.1001"),
    ("ZipVoice fine-tune best", "0.08253 at checkpoint-20"),
    ("ZipVoice-Distill start", "0.05546"),
    ("ZipVoice-Distill best", "0.03429 at checkpoint-10"),
]


def copy_audio() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    for stale in AUDIO.glob("*.wav*"):
        stale.unlink()
    for variant in [*VARIANTS, *LOW_STEP_VARIANTS]:
        for line in TEST_LINES:
            src = variant["src"] / line[variant["file_key"]]
            dst = AUDIO / f"{variant['id']}_{line['id']}.wav"
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)


def audio_src(variant_id: str, line_id: str) -> str:
    return f"audio/zipvoice_distill_v1/{variant_id}_{line_id}.wav"


def esc(value: object) -> str:
    return html.escape(str(value))


def model_card(variant: dict[str, str]) -> str:
    return f"""
    <article class="model-card">
      <div class="tag">{esc(variant['tag'])}</div>
      <h2>{esc(variant['name'])}</h2>
      <dl>
        <div><dt>Model</dt><dd>{esc(variant['model'])}</dd></div>
        <div><dt>一句話生成</dt><dd>{esc(variant['one_line'])}</dd></div>
        <div><dt>模型大小</dt><dd>{esc(variant['size'])}</dd></div>
        <div><dt>峰值記憶體</dt><dd>{esc(variant['memory'])}</dd></div>
        <div><dt>冷啟動參考</dt><dd>{esc(variant['time'])}</dd></div>
        <div><dt>速度判斷</dt><dd>{esc(variant['rtf'])}</dd></div>
      </dl>
      <p>{esc(variant['quality'])}</p>
    </article>
    """


def sample_row(line: dict[str, str]) -> str:
    cells = []
    for variant in VARIANTS:
        cells.append(
            f"""
            <section class="sample-card">
              <div class="sample-head">
                <b>{esc(variant['name'])}</b>
                <span>{esc(variant['tag'])}</span>
              </div>
              <audio controls preload="metadata" src="{esc(audio_src(variant['id'], line['id']))}"></audio>
            </section>
            """
        )
    return f"""
    <article class="line-block">
      <p class="line-text">{esc(line['text'])}</p>
      <div class="sample-grid">{''.join(cells)}</div>
    </article>
    """


def low_step_row(line: dict[str, str]) -> str:
    cells = []
    for variant in LOW_STEP_VARIANTS:
        cells.append(
            f"""
            <section class="sample-card low-step-card">
              <div class="sample-head">
                <b>{esc(variant['name'])}</b>
                <span>{esc(variant['tag'])}</span>
              </div>
              <audio controls preload="metadata" src="{esc(audio_src(variant['id'], line['id']))}"></audio>
              <dl class="mini-metrics">
                <div><dt>RTF</dt><dd>{esc(variant['rtf'])}</dd></div>
                <div><dt>Peak</dt><dd>{esc(variant['memory'])}</dd></div>
              </dl>
              <p>{esc(variant['note'])}</p>
            </section>
            """
        )
    return f"""
    <article class="line-block">
      <p class="line-text">{esc(line['text'])}</p>
      <div class="low-step-grid">{''.join(cells)}</div>
    </article>
    """


def build_html() -> str:
    copy_audio()
    loss_items = "".join(
        f"<li><b>{esc(k)}</b><span>{esc(v)}</span></li>" for k, v in LOSSES
    )
    summary = {
        "report": str(OUT / "distillation_v1.html"),
        "variants": [
            {
                "id": v["id"],
                "name": v["name"],
                "size": v["size"],
                "memory": v["memory"],
                "one_line": v["one_line"],
                "time": v["time"],
            }
            for v in VARIANTS
        ],
        "low_step_variants": [
            {
                "id": v["id"],
                "name": v["name"],
                "rtf": v["rtf"],
                "memory": v["memory"],
                "note": v["note"],
            }
            for v in LOW_STEP_VARIANTS
        ],
        "losses": LOSSES,
    }
    (OUT / "zipvoice_distill_v1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ZipVoice 完整蒸餾 v1</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #17181c;
      --muted: #5f6977;
      --line: #d9dee8;
      --accent: #b0182b;
      --accent-soft: #fff1f3;
      --soft: #f2f5f8;
      --ok: #1c6b53;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      line-height: 1.58;
    }}
    header, main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px 14px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 36px;
      line-height: 1.12;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 6px 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    p {{ margin: 0; }}
    .lead {{
      color: var(--muted);
      font-size: 16px;
      max-width: 840px;
    }}
    .answer {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }}
    .answer div, .model-card, .line-block, .note-panel, .step {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .answer b {{
      display: block;
      color: var(--accent);
      margin-bottom: 4px;
    }}
    .model-grid, .sample-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .low-step-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .tag {{
      color: var(--accent);
      font-weight: 800;
      font-size: 13px;
    }}
    dl {{
      margin: 0;
      display: grid;
      gap: 8px;
    }}
    dt {{
      color: var(--muted);
      font-size: 12px;
    }}
    dd {{
      margin: 2px 0 0;
      font-weight: 680;
      overflow-wrap: anywhere;
    }}
    .model-card p {{
      margin-top: 12px;
      color: var(--muted);
    }}
    .note-panel {{
      margin-top: 14px;
    }}
    .note-panel ul {{
      margin: 8px 0 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 6px;
    }}
    .note-panel li {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 6px;
    }}
    .note-panel li:last-child {{ border-bottom: 0; }}
    .note-panel span {{
      color: var(--muted);
      text-align: right;
    }}
    .line-block {{
      margin-top: 12px;
    }}
    .line-text {{
      font-size: 17px;
      font-weight: 750;
      margin-bottom: 10px;
    }}
    .sample-card {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    .low-step-card p {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }}
    .mini-metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }}
    .mini-metrics div {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px;
    }}
    .sample-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 8px;
      font-size: 13px;
    }}
    .sample-head span {{
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }}
    audio {{
      width: 100%;
      height: 38px;
    }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .step strong {{
      display: block;
      color: var(--ok);
      margin-bottom: 4px;
    }}
    .callout {{
      background: var(--accent-soft);
      border-color: #efc8cf;
    }}
    @media (max-width: 820px) {{
      header, main {{ padding: 16px 12px; }}
      h1 {{ font-size: 30px; }}
      .lead, .line-text {{ font-size: 16px; }}
      .answer, .model-grid, .sample-grid, .low-step-grid, .steps {{ grid-template-columns: 1fr; }}
      .model-card, .line-block, .note-panel, .step {{ padding: 12px; }}
      .note-panel li {{ display: block; }}
      .note-panel span {{ display: block; text-align: left; margin-top: 2px; }}
      .sample-head {{ display: block; }}
      .sample-head span {{ display: block; text-align: left; margin-top: 2px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ZipVoice 完整蒸餾 v1</h1>
    <p class="lead">這頁保留 Qwen3 1.7B 老師、目前 ZipVoice-Distill ONNX int8、6-step 候選，並新增 2/3/4-step 大刀版比較。每一列都是同一句台詞，方便直接聽像不像、穩不穩。</p>
    <div class="answer">
      <div><b>有機會更小嗎</b><p>有，但不是靠把 steps 調低。現在 int8 包約 177MB；要更小要換小 vocoder、剪枝/低秩、或訓練單聲音小架構。</p></div>
      <div><b>有機會更快嗎</b><p>有。2/3/4-step 已跑出來，RTF 約 0.11 / 0.15 / 0.20；速度有砍下去，但要用聽感確認掉字。</p></div>
      <div><b>已經蒸餾完了嗎</b><p>第一輪完整蒸餾已完成；第二輪 few-step 蒸餾已做到早期 checkpoint-10，先拿來實測低步數。</p></div>
    </div>
  </header>
  <main>
    <section class="model-grid">{''.join(model_card(v) for v in VARIANTS)}</section>

    <section class="note-panel">
      <h2>訓練結果</h2>
      <p>loss 有下降，代表模型確實在吸收老師語料；但 TTS 最後仍要靠同文試聽檢查音色、尾音和掉字。</p>
      <ul>{loss_items}</ul>
    </section>

    <section class="note-panel callout">
      <h2>生成時間怎麼看</h2>
      <p>決策先看「一句話生成」：這比較接近 app 載入模型後，按下播放要等多久。冷啟動 command 只用來估第一次打開 app、初始化 ONNX session 和 vocoder 的成本。這次 2-step 平均 RTF 約 0.107、3-step 約 0.151、4-step 約 0.199-0.213、6-step 約 0.30-0.31、8-step 約 0.40-0.41。可以更快，但 steps 越低越容易掉字或語氣變粗；現在要在 3-step 和 4-step 中挑手機 demo 預設。</p>
    </section>

    <section class="note-panel callout">
      <h2>2-4 step 大刀版</h2>
      <p>Stage2 few-step checkpoint-10 沒有讓同 step 的速度本質變快，因為架構一樣；它要證明的是低 step 聽感能不能比 Stage1 穩。峰值記憶體仍約 1.22GB，後續要降記憶體要處理 decoder/vocoder/session，不是只調 steps。</p>
    </section>

    {''.join(low_step_row(line) for line in TEST_LINES)}

    <section class="note-panel callout">
      <h2>下一步縮小方向</h2>
      <p>目前 6-step 是更快，不是更小。真正要把手機峰值壓下來，優先順序是：只打包 int8 檔案、把 52MB vocoder 換成更小或量化版、對 119MB decoder 做結構剪枝/低秩、最後才是重新訓練單一聲音專用小模型。目標會是中階手機可接受的峰值記憶體，而不是只看檔案大小。</p>
    </section>

    {''.join(sample_row(line) for line in TEST_LINES)}

    <section class="note-panel">
      <h2>蒸餾過程教學</h2>
      <div class="steps">
        <div class="step"><strong>1. 固定老師聲音</strong><p>先選定台灣國語低卷舌女生 prompt，讓 Qwen3 1.7B 成為目標聲音。</p></div>
        <div class="step"><strong>2. 產老師語料</strong><p>用同一個 prompt 產 500 句，總長約 1.25 小時，內容涵蓋短句、安撫、疑問和長句。</p></div>
        <div class="step"><strong>3. 微調 ZipVoice</strong><p>把老師語料整理成 ZipVoice TSV/fbank，從官方 ZipVoice checkpoint 開始 fine-tune，先讓模型靠近老師音色。</p></div>
        <div class="step"><strong>4. 訓練 ZipVoice-Distill</strong><p>用 fine-tuned ZipVoice 當 teacher，訓練 ZipVoice-Distill student，讓 student 用更少步數生成。</p></div>
        <div class="step"><strong>5. 匯出 ONNX int8</strong><p>把 checkpoint-10 匯出成 text encoder 和 decoder ONNX，再做 dynamic int8，得到手機候選包。</p></div>
        <div class="step"><strong>6. 調 steps 與聽感</strong><p>8-step 比較穩，6-step 比較快。接下來要用你聽感挑 checkpoint，再決定要不要長訓練或壓模型。</p></div>
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "distillation_v1.html").write_text(build_html(), encoding="utf-8")
    print(OUT / "distillation_v1.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
