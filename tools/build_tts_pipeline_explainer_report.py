#!/usr/bin/env python3
"""Build a phone-readable explainer for Qwen/Cosy to ZipVoice distillation."""

from __future__ import annotations

import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
REPORT_DIR = BASE / "reports" / "qwen_cosy_zipvoice_pipeline_v1"
OUT = REPORT_DIR / "qwen_cosy_zipvoice_pipeline_v1.html"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen / Cosy 到 ZipVoice 蒸餾圖解</title>
  <style>
    :root {{
      --bg: #f5f2ec;
      --paper: #fffefa;
      --ink: #24211d;
      --muted: #6b6258;
      --line: #d8d0c5;
      --red: #b23b38;
      --blue: #255f7f;
      --green: #3f7659;
      --soft: #eee5da;
      --warn: #8b6223;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      font-size: 16px;
      line-height: 1.62;
    }}
    main {{
      width: min(1060px, 100%);
      margin: 0 auto;
      padding: 20px 14px 56px;
    }}
    .eyebrow {{
      color: var(--red);
      font-size: 13px;
      font-weight: 800;
    }}
    h1 {{
      margin: 6px 0 10px;
      font-size: clamp(30px, 8vw, 48px);
      line-height: 1.08;
      letter-spacing: 0;
    }}
    h2 {{ margin: 0 0 10px; font-size: 23px; line-height: 1.25; }}
    h3 {{ margin: 0 0 8px; font-size: 17px; }}
    p {{ margin: 7px 0; }}
    .lead {{ color: var(--muted); max-width: 780px; }}
    section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 14px 0;
    }}
    .two {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .lane {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fffaf2;
    }}
    .flow {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      margin-top: 10px;
    }}
    .box {{
      border: 1px solid var(--line);
      border-left: 5px solid var(--blue);
      border-radius: 8px;
      padding: 10px;
      background: var(--paper);
    }}
    .box.teacher {{ border-left-color: var(--red); }}
    .box.student {{ border-left-color: var(--green); }}
    .box.speed {{ border-left-color: var(--warn); }}
    .arrow {{
      text-align: center;
      color: var(--muted);
      font-weight: 800;
      line-height: 1;
    }}
    .tag {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      color: var(--muted);
      background: var(--soft);
      font-size: 12px;
      margin: 2px 4px 2px 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .metric b {{
      display: block;
      color: var(--blue);
      font-size: 25px;
      line-height: 1.15;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 13px;
    }}
    ul {{
      padding-left: 20px;
      margin: 8px 0;
    }}
    li {{ margin: 5px 0; }}
    code {{
      background: #eee4d8;
      border-radius: 5px;
      padding: 2px 5px;
      word-break: break-word;
    }}
    .path {{
      color: var(--muted);
      word-break: break-word;
      font-size: 12px;
    }}
    @media (max-width: 800px) {{
      main {{ padding-inline: 10px; }}
      .two, .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">TTS distillation map</div>
      <h1>Qwen / Cosy 到 ZipVoice 做了什麼</h1>
      <p class="lead">白話講：大模型負責產生你喜歡的「老師聲音」，ZipVoice 負責學成比較小、比較能放進手機的「學生模型」。真正困難點不是只讓聲音像，而是要在 4-step 甚至 3-step 還不要壞掉。</p>
    </header>

    <section>
      <h2>兩條路線對照</h2>
      <div class="two">
        <div class="lane">
          <h3>之前：Qwen 1.7B → ZipVoice</h3>
          <div class="tag">已做過</div><div class="tag">重點是 prompt 聲音</div>
          <div class="flow">
            <div class="box teacher"><strong>Qwen 1.7B 老師</strong><p>用「台湾国语低卷舌 / 台大女生 / 清亮温柔」這類 voice prompt，先找到你覺得對的聲音。</p></div>
            <div class="arrow">↓</div>
            <div class="box"><strong>老師語音樣本</strong><p>生成多句台詞，拿來測試 ZipVoice 是否能模仿這個風格。</p></div>
            <div class="arrow">↓</div>
            <div class="box student"><strong>ZipVoice 學生</strong><p>用 ZipVoice / Sherpa-ONNX int8 當手機候選，測 8、6、4、3、2 step。2-step 不穩，3/4-step 是速度候選。</p></div>
            <div class="arrow">↓</div>
            <div class="box speed"><strong>手機方向</strong><p>ONNX/int8、模型載入一次、降低 step、縮短 prompt/reference、避開 Python runtime。</p></div>
          </div>
        </div>

        <div class="lane">
          <h3>現在：CosyVoice2 → ZipVoice</h3>
          <div class="tag">正在做</div><div class="tag">重點是語料品質</div>
          <div class="flow">
            <div class="box teacher"><strong>CosyVoice2 golden teacher</strong><p>先從女聲 reference 裡挑一個最像「台灣溫柔漂亮女生」的 reference。你目前覺得 prompt reference 比某些 golden sample 更穩。</p></div>
            <div class="arrow">↓</div>
            <div class="box"><strong>500 句老師語料</strong><p>用同一個 reference 生成 500 句日常句。下一步先讓你篩選，壞句不要拿去教學生。</p></div>
            <div class="arrow">↓</div>
            <div class="box student"><strong>ZipVoice fine-tune</strong><p>已做一版 decoder fine-tune 600 iterations，先聽 16-step 完整複製效果，再看 8/4-step 下降後是否還像。</p></div>
            <div class="arrow">↓</div>
            <div class="box speed"><strong>真正 few-step distillation</strong><p>如果 4-step 只是硬降推論 step 會壞，下一步要用 16-step/teacher 軌跡去教 4-step student，而不是只把 step 數調小。</p></div>
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2>現在已完成到哪裡</h2>
      <div class="grid">
        <div class="metric"><b>500</b><span>Cosy 老師句子已生成</span></div>
        <div class="metric"><b>468 / 32</b><span>ZipVoice train / dev manifest</span></div>
        <div class="metric"><b>600 iters</b><span>第一版 ZipVoice decoder fine-tune</span></div>
      </div>
      <p>但你剛剛抓到一個關鍵：不是每句文字的聽感都穩，而且 golden teacher sample 不一定比 prompt reference 好。所以現在最值得做的是先篩 500 句，保留真正像、字又準的句子，再做下一輪更穩的 ZipVoice。</p>
    </section>

    <section>
      <h2>step 是什麼意思</h2>
      <p>ZipVoice 這類 flow-matching TTS 不是一次吐出聲音，而是用多次修正把聲音從粗到細生成出來。step 就是修正次數。</p>
      <ul>
        <li><strong>16-step：</strong>品質上限檢查，通常比較像、比較穩，但慢。</li>
        <li><strong>8-step：</strong>折衷版，品質可能還可以，速度約快一半。</li>
        <li><strong>4-step：</strong>手機候選，但很容易掉字、音色變、節奏怪。</li>
        <li><strong>2-step：</strong>之前測過不可用，聲音壞太明顯。</li>
      </ul>
      <p>真正蒸餾的重點，是讓 4-step 學會接近 16-step 的結果，而不是單純把 16-step 模型拿去只跑 4-step。</p>
    </section>

    <section>
      <h2>Qwen 那版還能不能更快</h2>
      <p>可以，但不是無限快。之前 Qwen→ZipVoice 的模型大小已經降很多，生成時間沒等比例下降，原因是 CPU 推論還有 step loop、vocoder、文字/音素處理、ONNX runtime overhead，不只是權重大小。</p>
      <ul>
        <li><strong>最有效：</strong>用真正 4-step / 3-step distillation，不只是推論時硬降 step。</li>
        <li><strong>很有效：</strong>正式走 Sherpa-ONNX C++/mobile runtime，不用 Python ONNX 測試的 peak RSS 當最終結論。</li>
        <li><strong>有幫助：</strong>模型載入一次、reference embedding/fbank/token cache 起來，按鈕只換文字。</li>
        <li><strong>有幫助：</strong>縮短 reference prompt，不要每句都重新跑 speaker prompt。</li>
        <li><strong>有風險但可測：</strong>int4 / mixed precision / smaller vocoder。這可能讓聲音變薄或字跑掉，所以要用相同句子 A/B。</li>
      </ul>
      <p>短結論：Qwen 那條路還能加快，但如果目標是手機即時，下一個真正大刀是「few-step 蒸餾 + mobile C++ runtime + cache speaker」，不是再盲目壓權重而已。</p>
    </section>

    <section>
      <h2>接下來 Cosy→ZipVoice 的正確做法</h2>
      <ul>
        <li>用 500 句挑選表先篩老師語料，只留像、準、乾淨的句子。</li>
        <li>對被標成「重生」的句子，用同一個好 reference 重新生成，不把壞樣本混進訓練。</li>
        <li>用精選 corpus 再 fine-tune ZipVoice，先聽 16-step 是否像老師。</li>
        <li>16-step 穩了之後，才做 8/4/3-step distillation 和 ONNX/int8 匯出。</li>
        <li>最後報告只比較：Cosy teacher、ZipVoice 16-step 完整複製、ZipVoice 4-step/3-step 加速版、手機 ONNX 版。</li>
      </ul>
      <p class="path">本頁：{esc(str(OUT))}</p>
    </section>
  </main>
</body>
</html>
"""
    OUT.write_text(html_text, encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
