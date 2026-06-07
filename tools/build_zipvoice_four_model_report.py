#!/usr/bin/env python3
"""Build a standalone four-model ZipVoice distillation report."""

from __future__ import annotations

import base64
import html
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
OUT = BASE / "reports" / "zipvoice_four_model"
ZIP_EGS = ROOT / "external" / "ZipVoice" / "egs" / "zipvoice"

LINES = [
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

MODELS = [
    {
        "id": "qwen_teacher",
        "name": "Qwen3 1.7B 老師",
        "role": "目標聲音",
        "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
        "steps": "不適用",
        "size": "約 2.2GB cache",
        "memory": "約 2.20GB peak RSS",
        "time": "約 1.70s / 句",
        "rtf": "老師基準",
        "src": OUT.parent / "zipvoice_three_way" / "teacher_qwen3_1p7b_compare" / "audio",
        "file_key": "teacher_file",
        "note": "聲音最符合目標：台灣國語、低卷舌、清亮溫柔。缺點是太大，手機離線部署壓力高。",
    },
    {
        "id": "zipvoice_original",
        "name": "ZipVoice 原始",
        "role": "未蒸餾 baseline",
        "model": "official ZipVoice base，ONNX dynamic int8",
        "steps": "16-step",
        "size": "text encoder 5.3MB + decoder 119MB + vocoder 52MB = 約 176MB",
        "memory": "1.63GB max RSS；Python ONNX CPU 測試",
        "time": "約 9.8s / 句；冷啟動 command 32.27s / 3 句",
        "rtf": "平均 RTF 1.4256",
        "src": ZIP_EGS / "results" / "zipvoice_original_onnx_int8_step16_th4",
        "file_key": "zip_file",
        "note": "沒有吃 Qwen 老師語料。檔案大小跟蒸餾版相近，但 16-step 推論慢很多，也不是目標台灣女生聲線。",
    },
    {
        "id": "distill_step3",
        "name": "ZipVoice-Distill 3-step",
        "role": "目前主候選",
        "model": "Stage2 few-step epoch-10，ONNX dynamic int8",
        "steps": "3-step",
        "size": "text encoder 5.3MB + decoder 119MB + vocoder 52MB = 約 176MB",
        "memory": "1.22GB max RSS；Python ONNX CPU 測試",
        "time": "live preload 約 0.97s / 句；batch 平均 RTF 0.1505",
        "rtf": "平均 RTF 0.1505",
        "src": ZIP_EGS / "results" / "qwen_teacher_stage2_fewstep10_onnx_int8_step3_th4",
        "file_key": "zip_file",
        "note": "速度已經有手機 demo 體感。若你覺得咬字可接受，它就是目前最適合繼續打磨的預設。",
    },
    {
        "id": "distill_step4",
        "name": "ZipVoice-Distill 4-step",
        "role": "較穩候選",
        "model": "Stage2 few-step epoch-10，ONNX dynamic int8",
        "steps": "4-step",
        "size": "text encoder 5.3MB + decoder 119MB + vocoder 52MB = 約 176MB",
        "memory": "1.22GB max RSS；Python ONNX CPU 測試",
        "time": "live preload 約 1.25s / 句；batch 平均 RTF 0.1990",
        "rtf": "平均 RTF 0.1990",
        "src": ZIP_EGS / "results" / "qwen_teacher_stage2_fewstep10_onnx_int8_step4_th4",
        "file_key": "zip_file",
        "note": "比 3-step 慢一點，但通常會比較穩。若 3-step 有字跑掉，4-step 是更安全的手機預設。",
    },
]


def esc(value: object) -> str:
    return html.escape(str(value))


def audio_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{data}"


def model_card(model: dict[str, str]) -> str:
    return f"""
    <article class="model-card">
      <div class="role">{esc(model['role'])}</div>
      <h2>{esc(model['name'])}</h2>
      <dl>
        <div><dt>Model</dt><dd>{esc(model['model'])}</dd></div>
        <div><dt>Steps</dt><dd>{esc(model['steps'])}</dd></div>
        <div><dt>模型大小</dt><dd>{esc(model['size'])}</dd></div>
        <div><dt>峰值記憶體</dt><dd>{esc(model['memory'])}</dd></div>
        <div><dt>生成時間</dt><dd>{esc(model['time'])}</dd></div>
        <div><dt>RTF</dt><dd>{esc(model['rtf'])}</dd></div>
      </dl>
      <p>{esc(model['note'])}</p>
    </article>
    """


def sample_row(line: dict[str, str]) -> str:
    cells = []
    for model in MODELS:
        src = Path(model["src"]) / line[model["file_key"]]
        if not src.exists():
            raise FileNotFoundError(src)
        cells.append(
            f"""
            <section class="sample-card">
              <div class="sample-head">
                <b>{esc(model['name'])}</b>
                <span>{esc(model['steps'])}</span>
              </div>
              <audio controls preload="metadata" src="{audio_data_uri(src)}"></audio>
            </section>
            """
        )
    return f"""
    <article class="line-block">
      <p class="line-text">{esc(line['text'])}</p>
      <div class="sample-grid">{''.join(cells)}</div>
    </article>
    """


def build_html() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "title": "Qwen 1.7B 到 ZipVoice-Distill 手機候選報告",
        "models": [
            {
                "id": m["id"],
                "name": m["name"],
                "steps": m["steps"],
                "size": m["size"],
                "memory": m["memory"],
                "time": m["time"],
                "rtf": m["rtf"],
            }
            for m in MODELS
        ],
        "excluded": "2-step 不採用：速度約 0.87s/句，但聽感/咬字穩定性不足。",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen 到 ZipVoice-Distill 四模型報告</title>
  <style>
    :root {{
      --bg: #f8f9fb;
      --panel: #ffffff;
      --ink: #17181c;
      --muted: #5e6876;
      --line: #dbe1ea;
      --accent: #b0182b;
      --soft: #f2f5f8;
      --ok: #17654f;
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
      font-size: 34px;
      line-height: 1.15;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 4px 0 12px;
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
      max-width: 900px;
    }}
    .summary {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .summary div, .model-card, .note-panel, .line-block, .step {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .summary b {{
      display: block;
      color: var(--accent);
      margin-bottom: 4px;
    }}
    .model-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .role {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 800;
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
      margin-top: 12px;
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
      background: #fff2f4;
      border-color: #efc8cf;
    }}
    .line-block {{
      margin-top: 12px;
    }}
    .line-text {{
      font-size: 17px;
      font-weight: 750;
      margin-bottom: 10px;
    }}
    .sample-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .sample-card {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
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
    @media (max-width: 900px) {{
      h1 {{ font-size: 29px; }}
      header, main {{ padding: 16px 12px; }}
      .summary, .model-grid, .sample-grid, .steps {{ grid-template-columns: 1fr; }}
      .summary div, .model-card, .note-panel, .line-block, .step {{ padding: 12px; }}
      .sample-head {{ display: block; }}
      .sample-head span {{ display: block; text-align: left; margin-top: 2px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Qwen 1.7B 到 ZipVoice-Distill 手機候選報告</h1>
    <p class="lead">這份只保留四個比較對象：Qwen3 1.7B 老師、ZipVoice 原始、蒸餾後 3-step、蒸餾後 4-step。2-step 已從主線移除，因為雖然最快，但目前聽感與咬字穩定性不足。</p>
    <section class="summary">
      <div><b>最快可用候選</b><p>3-step，live preload 約 0.97s / 句，適合先做手機 demo 預設。</p></div>
      <div><b>較穩候選</b><p>4-step，live preload 約 1.25s / 句，若 3-step 有漏字就改用它。</p></div>
      <div><b>為何不選 2-step</b><p>2-step 約 0.87s / 句，但聽感不可靠；不放進主比較。</p></div>
    </section>
  </header>
  <main>
    <section class="model-grid">{''.join(model_card(m) for m in MODELS)}</section>

    <section class="note-panel callout">
      <h2>Step 是什麼意思</h2>
      <p>ZipVoice 這類 flow-matching TTS 不是一次直接吐出聲音，而是從雜訊一步一步修成語音特徵。step 就是這個修正迭代次數。原始 ZipVoice 預設 16-step，所以慢；ZipVoice-Distill 的目標是讓學生模型學會用更少步數也產生可聽語音。steps 越少通常越快，但太低會更容易漏字、糊字、聲音變粗或不穩。</p>
    </section>

    <section class="note-panel">
      <h2>完整蒸餾怎麼做</h2>
      <div class="steps">
        <div class="step"><strong>1. 固定老師聲音</strong><p>先選定 Qwen3 1.7B VoiceDesign prompt：台灣國語低卷舌、年輕女生、清亮溫柔、有書卷氣。這就是後面要模仿的聲線。</p></div>
        <div class="step"><strong>2. 產老師語料</strong><p>用 Qwen 老師產約 500 句，形成第一批台灣女生聲線 corpus。這批語料包含短句、安撫句、推理句和長句。</p></div>
        <div class="step"><strong>3. 微調 ZipVoice</strong><p>把老師音檔轉成 ZipVoice 訓練格式與 fbank，先讓官方 ZipVoice 往老師音色靠近，得到一個 teacher-adapted ZipVoice。</p></div>
        <div class="step"><strong>4. 訓練 ZipVoice-Distill</strong><p>再用 ZipVoice-Distill student 學這個 teacher，重點不是把檔案變小，而是把生成步數從原始 16-step 壓到 3/4-step。</p></div>
        <div class="step"><strong>5. few-step 二階段</strong><p>針對 2/3/4-step 做第二階段低步數實驗。2-step 太不穩，所以目前只留下 3-step 與 4-step。</p></div>
        <div class="step"><strong>6. 匯出手機路線</strong><p>把 student 匯出成 ONNX，再做 dynamic int8。現在檔案約 176MB，峰值記憶體仍要靠 runtime、vocoder 和 session 管理繼續壓。</p></div>
      </div>
    </section>

    {''.join(sample_row(line) for line in LINES)}

    <section class="note-panel callout">
      <h2>目前結論</h2>
      <p>蒸餾的主要收益已經出現：原始 ZipVoice 16-step 在 Python ONNX 測試約 9.8s / 句，蒸餾 3-step 約 0.97s / 句、4-step 約 1.25s / 句。檔案大小還沒有變小，因為架構仍是同級 ZipVoice；下一刀要砍記憶體與模型體積，就要做 decoder/vocoder 瘦身，而不是只調 steps。</p>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    html_path = OUT / "four_model_distillation_report.html"
    html_path.write_text(build_html(), encoding="utf-8")
    shutil.copy2(html_path, OUT / "index.html")
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
