from __future__ import annotations

import html
import json
import shutil
from pathlib import Path


ROOT = Path("/Users/ader/Documents/App")
OUT = ROOT / "distillation/taiwan_mandarin_low_r/reports/target_route_deep_technical_v1"
ASSETS = OUT / "assets"


def copy_asset(src: str, dest: str) -> str:
    src_path = ROOT / src
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    ASSETS.mkdir(parents=True, exist_ok=True)
    dst_path = ASSETS / dest
    shutil.copy2(src_path, dst_path)
    return f"assets/{dest}"


def load_json(rel: str):
    with (ROOT / rel).open("r", encoding="utf-8") as f:
        return json.load(f)


def audio_card(title: str, text: str, src: str, note: str = "") -> str:
    return f"""
      <article class="sample">
        <div>
          <b>{html.escape(title)}</b>
          <p>{html.escape(text)}</p>
          {f'<small>{html.escape(note)}</small>' if note else ''}
        </div>
        <audio controls preload="metadata" src="{html.escape(src)}"></audio>
      </article>
    """


def sample_grid(samples: list[dict]) -> str:
    return '<div class="samples">' + "\n".join(audio_card(**s) for s in samples) + "</div>"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    qwen_manifest = load_json("distillation/taiwan_mandarin_low_r/teacher_qwen3_1p7b_seed/manifest.json")
    cosy_manifest = load_json("distillation/taiwan_mandarin_low_r/teacher_cosy_clear_best2_golden_daily_500_v1/manifest.json")
    zip_runtime = load_json("distillation/taiwan_mandarin_low_r/reports/zipvoice_three_way/runtime_metrics.json")
    zip_distill = load_json("distillation/taiwan_mandarin_low_r/reports/zipvoice_three_way/zipvoice_distill_v1_summary.json")
    speedup = load_json("distillation/taiwan_mandarin_low_r/reports/qwen_zipvoice_speedup_v1/summary.json")
    cosy_zip_manifest = load_json("distillation/taiwan_mandarin_low_r/datasets/zipvoice_cosy_golden_daily_500_v1/manifest.json")
    cosy_student_summary = load_json("distillation/taiwan_mandarin_low_r/reports/zipvoice_student_600_audition_v1/summary.json")

    qwen_samples = [
        {
            "title": "Qwen 1.7B 原型 01",
            "text": qwen_manifest[0]["text"],
            "src": copy_asset("distillation/taiwan_mandarin_low_r/teacher_qwen3_1p7b_seed/audio/seed_0001.wav", "qwen_seed_0001.wav"),
            "note": "VoiceDesign prompt: taiwan_mandarin_low_r",
        },
        {
            "title": "Qwen 1.7B 原型 02",
            "text": qwen_manifest[1]["text"],
            "src": copy_asset("distillation/taiwan_mandarin_low_r/teacher_qwen3_1p7b_seed/audio/seed_0002.wav", "qwen_seed_0002.wav"),
            "note": "同一聲音 profile，換句子驗穩定度",
        },
        {
            "title": "Qwen 1.7B 原型 03",
            "text": qwen_manifest[2]["text"],
            "src": copy_asset("distillation/taiwan_mandarin_low_r/teacher_qwen3_1p7b_seed/audio/seed_0003.wav", "qwen_seed_0003.wav"),
            "note": "低卷舌、台灣國語檢查句",
        },
    ]

    cosy_samples = [
        {
            "title": "Cosy zero-shot 老師 01",
            "text": cosy_manifest[0]["text"],
            "src": copy_asset("distillation/taiwan_mandarin_low_r/teacher_cosy_clear_best2_golden_daily_500_v1/audio/golden_daily_0001.wav", "cosy_golden_daily_0001.wav"),
            "note": "CosyVoice2-0.5B + clear_best2_7s reference",
        },
        {
            "title": "Cosy zero-shot 老師 02",
            "text": cosy_manifest[1]["text"],
            "src": copy_asset("distillation/taiwan_mandarin_low_r/teacher_cosy_clear_best2_golden_daily_500_v1/audio/golden_daily_0002.wav", "cosy_golden_daily_0002.wav"),
            "note": "同 reference，日常對話句",
        },
        {
            "title": "Cosy zero-shot 老師 03",
            "text": cosy_manifest[2]["text"],
            "src": copy_asset("distillation/taiwan_mandarin_low_r/teacher_cosy_clear_best2_golden_daily_500_v1/audio/golden_daily_0003.wav", "cosy_golden_daily_0003.wav"),
            "note": "同 reference，台灣日常語氣",
        },
    ]

    qwen_zip_samples = [
        {
            "title": "ZipVoice 學 Qwen 01",
            "text": "Qwen teacher corpus 的 smoke 句，ZipVoice int8 / 8-step 學同聲線輸出。",
            "src": copy_asset("distillation/taiwan_mandarin_low_r/reports/zipvoice_three_way/audio/zipvoice_distill_v1/zipvoice_int8_step8_smoke_01.wav", "qwen_zipvoice_step8_01.wav"),
            "note": "約 177MB ONNX int8；早期品質版 8-step",
        },
        {
            "title": "ZipVoice 學 Qwen 02",
            "text": "同一 Qwen teacher route 的第二句，檢查咬字與音色一致性。",
            "src": copy_asset("distillation/taiwan_mandarin_low_r/reports/zipvoice_three_way/audio/zipvoice_distill_v1/zipvoice_int8_step8_smoke_02.wav", "qwen_zipvoice_step8_02.wav"),
            "note": "8-step 比 3/4-step 穩，但較慢",
        },
        {
            "title": "ZipVoice 學 Qwen 03",
            "text": "同一 Qwen teacher route 的第三句，檢查長句穩定性。",
            "src": copy_asset("distillation/taiwan_mandarin_low_r/reports/zipvoice_three_way/audio/zipvoice_distill_v1/zipvoice_int8_step8_smoke_03.wav", "qwen_zipvoice_step8_03.wav"),
            "note": "用來當後續 few-step student 的 teacher target",
        },
    ]

    cosy_zip16_samples = [
        {
            "title": "ZipVoice 學 Cosy 16-step 01",
            "text": "Cosy teacher route 的完整品質檢查：先看 16-step 能不能接近老師。",
            "src": copy_asset("external/ZipVoice/egs/zipvoice/results/cosy_true_distill_onnx_int8_step16/cosy_01.wav.wav", "cosy_zipvoice_step16_01.wav"),
            "note": "16-step 是品質上限檢查，不是手機速度目標",
        },
        {
            "title": "ZipVoice 學 Cosy 16-step 02",
            "text": "同一 Cosy teacher route 的第二句。",
            "src": copy_asset("external/ZipVoice/egs/zipvoice/results/cosy_true_distill_onnx_int8_step16/cosy_02.wav.wav", "cosy_zipvoice_step16_02.wav"),
            "note": "用來判斷 student 是否真的學到 Cosy 聲線",
        },
        {
            "title": "ZipVoice 學 Cosy 16-step 03",
            "text": "同一 Cosy teacher route 的第三句。",
            "src": copy_asset("external/ZipVoice/egs/zipvoice/results/cosy_true_distill_onnx_int8_step16/cosy_03.wav.wav", "cosy_zipvoice_step16_03.wav"),
            "note": "若 16-step 都不像，降 step 沒意義",
        },
    ]

    qwen_fewstep_samples = [
        {
            "title": "Qwen→ZipVoice few-step 3-step 01",
            "text": "真正要往手機即時靠的版本：3-step / Sherpa-ONNX runtime。",
            "src": copy_asset("distillation/taiwan_mandarin_low_r/reports/qwen_zipvoice_speedup_v1/audio/sherpa_step3_s01.wav", "qwen_zipvoice_sherpa_step3_01.wav"),
            "note": "已載入模型後 avg 約 1.11s/句；peak RSS 約 959MB",
        },
        {
            "title": "Qwen→ZipVoice few-step 3-step 02",
            "text": "同 route 第二句，檢查低 step 是否漏字或電子聲。",
            "src": copy_asset("distillation/taiwan_mandarin_low_r/reports/qwen_zipvoice_speedup_v1/audio/sherpa_step3_s02.wav", "qwen_zipvoice_sherpa_step3_02.wav"),
            "note": "3-step 是速度候選，不是保守品質版",
        },
        {
            "title": "Qwen→ZipVoice few-step 3-step 03",
            "text": "同 route 第三句，檢查穩定性。",
            "src": copy_asset("distillation/taiwan_mandarin_low_r/reports/qwen_zipvoice_speedup_v1/audio/sherpa_step3_s03.wav", "qwen_zipvoice_sherpa_step3_03.wav"),
            "note": "需要繼續用 few-step distillation 修乾淨度",
        },
    ]

    cosy_fewstep_samples = [
        {
            "title": "Cosy→ZipVoice 4-step 01",
            "text": "Cosy route 的 4-step 速度候選，聽聲音乾淨度與漏字。",
            "src": copy_asset("external/ZipVoice/egs/zipvoice/results/cosy_true_distill_onnx_int8_step4/cosy_01.wav.wav", "cosy_zipvoice_step4_01.wav"),
            "note": "硬降 step 會掉細節；需要真正 few-step distillation",
        },
        {
            "title": "Cosy→ZipVoice 4-step 02",
            "text": "同 route 第二句。",
            "src": copy_asset("external/ZipVoice/egs/zipvoice/results/cosy_true_distill_onnx_int8_step4/cosy_02.wav.wav", "cosy_zipvoice_step4_02.wav"),
            "note": "手機目標要接近這種 step 數，但音質要更穩",
        },
        {
            "title": "Cosy→ZipVoice 4-step 03",
            "text": "同 route 第三句。",
            "src": copy_asset("external/ZipVoice/egs/zipvoice/results/cosy_true_distill_onnx_int8_step4/cosy_03.wav.wav", "cosy_zipvoice_step4_03.wav"),
            "note": "目前可聽，但不是最終品質",
        },
    ]

    qwen_runtime = zip_runtime["teacher_qwen3_1p7b"]
    zip_teacher_ref = zip_runtime["zipvoice_teacher_ref"]
    speed_best = speedup["best_candidate"]
    losses = zip_distill["losses"]

    html_doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>紅色蝴蝶結 TTS 主線深度技術報告 v1｜桌機好讀版</title>
<style>
:root {{
  --paper:#f7f5ef; --ink:#25231f; --muted:#69645b; --line:#ddd6c8;
  --panel:#fffdf8; --red:#b7202f; --soft:#efe7d8; --code:#1e1e1e;
  --qwen:#e9f1ff; --cosy:#fff1e2; --zip:#f0f6e7; --ship:#f8e7ea;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font:16px/1.72 -apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif;
}}
main {{ max-width:1480px; margin:0 auto; padding:46px 34px 90px; }}
h1 {{ font-size:38px; line-height:1.16; margin:0 0 10px; letter-spacing:0; max-width:1100px; }}
h2 {{ font-size:24px; margin:42px 0 14px; border-top:1px solid var(--line); padding-top:28px; }}
h3 {{ font-size:18px; margin:24px 0 8px; }}
p {{ margin:8px 0 14px; }}
p, li, td, .notice, .card {{ overflow-wrap:anywhere; }}
.lead {{ font-size:18px; max-width:1080px; }}
.eyebrow {{ color:var(--red); font-weight:700; letter-spacing:.04em; text-transform:uppercase; font-size:12px; }}
.notice {{ background:#fff8eb; border:1px solid #ead8b7; padding:14px 16px; border-radius:8px; }}
.toc {{ display:flex; flex-wrap:wrap; gap:8px; margin:22px 0 20px; }}
.toc a {{ color:var(--ink); text-decoration:none; background:var(--panel); border:1px solid var(--line); border-radius:999px; padding:6px 12px; font-size:13px; }}
.toc a:hover {{ border-color:var(--red); color:var(--red); }}
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
.card b {{ color:var(--red); }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; font-size:15px; }}
th,td {{ text-align:left; vertical-align:top; padding:12px 14px; border-bottom:1px solid var(--line); }}
th {{ background:var(--soft); font-size:13px; }}
tr:last-child td {{ border-bottom:0; }}
code,kbd {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.92em; }}
pre {{
  background:var(--code); color:#f8f3e8; border-radius:8px; padding:14px 16px;
  overflow:auto; overflow-wrap:anywhere; white-space:pre-wrap; line-height:1.5; font-size:13px;
}}
.flow {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:18px 0; }}
.lane {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
.arrow {{ color:var(--muted); font-weight:700; padding:3px 0; }}
.route-map {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin:18px 0 12px; }}
.route-lane {{ border:1px solid var(--line); border-radius:10px; background:var(--panel); overflow:hidden; }}
.route-head {{ padding:14px 16px; font-weight:800; border-bottom:1px solid var(--line); }}
.route-head.qwen {{ background:var(--qwen); }}
.route-head.cosy {{ background:var(--cosy); }}
.route-body {{ padding:14px; display:grid; gap:10px; }}
.route-node {{ border:1px solid var(--line); background:#fff; border-radius:8px; padding:12px; }}
.route-node b {{ display:block; margin-bottom:4px; }}
.route-node small {{ color:var(--muted); }}
.route-arrow {{ text-align:center; color:var(--muted); font-weight:800; line-height:1; }}
.merge {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; align-items:stretch; margin-top:12px; }}
.merge-card {{ border:1px solid var(--line); background:var(--zip); border-radius:10px; padding:14px; }}
.merge-card.ship {{ background:var(--ship); }}
.compact-table td:first-child {{ width:18%; font-weight:700; }}
.compact-table td:nth-child(2) {{ width:30%; }}
.model-stack {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
.stack-card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
.stack-card h3 {{ margin-top:0; }}
.stack-card ul {{ margin:8px 0 0; padding-left:18px; }}
.samples {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin:14px 0 20px; }}
.sample {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; display:flex; flex-direction:column; gap:8px; min-height:210px; }}
.sample p {{ margin:5px 0; color:var(--muted); font-size:14px; }}
.sample small {{ display:block; color:#8a8173; font-size:12px; }}
audio {{ width:100%; margin-top:auto; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.muted {{ color:var(--muted); }}
.ok {{ color:#1f7a46; font-weight:700; }}
.warn {{ color:#a45b00; font-weight:700; }}
@media (max-width:900px) {{ .grid,.samples,.flow,.two,.route-map,.merge,.model-stack {{ grid-template-columns:1fr; }} main {{ padding:26px 14px 60px; }} }}
</style>
</head>
<body>
<main>
  <div class="eyebrow">target route only · Qwen/Q1 1.7B · CosyVoice2 · ZipVoice/Z-Voice</div>
  <h1>紅色蝴蝶結 TTS 主線深度技術報告 v1｜桌機好讀版</h1>
  <p class="lead">這份只講你要的主線：<b>Qwen/Q1 1.7B 跟 Cosy 是兩個獨立 teacher</b>，各自做聲音模仿；ZipVoice 再分別學 Qwen teacher corpus 與 Cosy teacher corpus；最後才討論 ZipVoice 自己的 distilled / few-step / ONNX int8 手機化。</p>

  <nav class="toc" aria-label="章節導覽">
    <a href="#summary">一頁結論</a>
    <a href="#roadmap">路線圖</a>
    <a href="#metrics">量測總表</a>
    <a href="#model-stack">模型內容物</a>
    <a href="#qwen-clone">Qwen clone</a>
    <a href="#qwen">Qwen 原理</a>
    <a href="#cosy">Cosy 原理</a>
    <a href="#flow">Flow Matching</a>
    <a href="#zip-qwen">ZipVoice 學 Qwen</a>
    <a href="#zip-cosy">ZipVoice 學 Cosy</a>
    <a href="#fewstep">Few-step 蒸餾</a>
    <a href="#cosy-sft">Cosy fine-tune 資料</a>
  </nav>

  <div class="notice">
    <b>重要校正：</b>這裡不是 <code>Qwen → Cosy → ZipVoice</code>。正確路線是雙分支：
    <code>Qwen/Q1 1.7B → ZipVoice</code> 與 <code>CosyVoice2 → ZipVoice</code>。Cosy 不是 Qwen 後面的一層；Cosy 自己就是另一個 teacher。
  </div>

  <h2 id="summary">一頁結論</h2>
  <div class="grid">
    <section class="card"><b>Qwen/Q1 1.7B</b><br>用 VoiceDesign prompt 直接設計「台灣國語低卷舌女生」聲音；速度不錯但模型大，主要價值是早期 teacher corpus。</section>
    <section class="card"><b>CosyVoice2</b><br>用 reference zero-shot clone 做另一條 teacher；目前音色上限最好，適合先做 server / fine-tune，再蒸餾到手機學生。</section>
    <section class="card"><b>ZipVoice 學老師</b><br>不是壓縮老師權重，而是用老師生成的 text/wav paired data 訓練 ZipVoice decoder，使 student acoustic distribution 靠近老師。</section>
    <section class="card"><b>手機終點</b><br>ZipVoice-Distill ONNX int8 + low steps + Sherpa runtime；真正目標是 3/4-step 還能乾淨可懂。</section>
  </div>

  <h2 id="roadmap">目前主線路線圖</h2>
  <div class="route-map">
    <section class="route-lane">
      <div class="route-head qwen">分支 A：Qwen/Q1 teacher，不接 Cosy</div>
      <div class="route-body">
        <div class="route-node"><b>1. VoiceDesign 原型</b><small>用文字 prompt 做台灣低卷舌女生聲音；這是設計，不是 reference clone。</small></div>
        <div class="route-arrow">↓</div>
        <div class="route-node"><b>2. Qwen teacher corpus</b><small>把選定聲音生成 500 句 text/wav，做早期 Qwen→ZipVoice 老師資料。</small></div>
        <div class="route-arrow">↓</div>
        <div class="route-node"><b>3. Qwen Base clone 待跑</b><small>改用 Qwen Base clone 模型，餵同一批女聲 dataset，和 Cosy 公平比較。</small></div>
      </div>
    </section>
    <section class="route-lane">
      <div class="route-head cosy">分支 B：Cosy teacher，reference clone</div>
      <div class="route-body">
        <div class="route-node"><b>1. Clean reference pack</b><small>用授權乾淨女聲 reference + transcript，做 Cosy zero-shot clone。</small></div>
        <div class="route-arrow">↓</div>
        <div class="route-node"><b>2. Cosy golden corpus</b><small>同一聲線生成 500 句日常對話，目前是 Cosy→ZipVoice 老師資料。</small></div>
        <div class="route-arrow">↓</div>
        <div class="route-node"><b>3. Cosy SFT 待做</b><small>補 10-30 分鐘以上乾淨資料，讓 Cosy 從 zero-shot 變成穩定 speaker。</small></div>
      </div>
    </section>
  </div>
  <div class="merge">
    <section class="merge-card"><b>共同學生：ZipVoice</b><br>分別學 Qwen corpus 與 Cosy corpus。先聽 16-step 確認像不像，再做真正 4/3-step distillation。</section>
    <section class="merge-card ship"><b>最後產品形態</b><br>桌機/雲端先用 teacher 保聲音上限；離線端才用 ZipVoice ONNX int8 + vocoder + cached prompt。</section>
  </div>

  <h2 id="metrics">量測總表</h2>
  <table class="compact-table">
    <thead><tr><th>模型</th><th>一句話理解</th><th>關鍵數字</th><th>目前怎麼用</th></tr></thead>
    <tbody>
      <tr>
        <td>Qwen VoiceDesign</td>
        <td>用文字描述「設計」一個台灣女生聲音，不是拿 reference 複製真人聲音。</td>
        <td>cache 約 2.2GB；peak RSS {html.escape(qwen_runtime["peak_rss_mb"])}；約 {html.escape(qwen_runtime["seconds_per_sentence"])} / 句。</td>
        <td>保留當聲音原型 / teacher baseline；真正 clone 要改跑 Qwen Base。</td>
      </tr>
      <tr>
        <td>CosyVoice2</td>
        <td>用 reference wav + transcript 做 zero-shot clone，目前聲音上限最好。</td>
        <td>model dir 約 4.5GB；舊報告 peak 約 5.73GB RSS；500 句 corpus 約 512MB。</td>
        <td>當主要 teacher；下一步是 Cosy speaker fine-tune。</td>
      </tr>
      <tr>
        <td>ZipVoice 學 Qwen</td>
        <td>用 Qwen 產出的 text/wav 訓練小一點的 flow TTS student。</td>
        <td>ONNX int8 約 177MB；peak 約 {html.escape(zip_teacher_ref["peak_rss_mb"])}；3-step avg {speed_best["avg_wall_seconds"]:.2f}s/句。</td>
        <td>速度路線；品質要繼續處理電子聲、漏字和 few-step。</td>
      </tr>
      <tr>
        <td>ZipVoice 學 Cosy</td>
        <td>用 Cosy golden corpus 訓練 ZipVoice，目標是把 Cosy 聲線搬到小模型。</td>
        <td>{html.escape(cosy_zip_manifest["train_count"].__str__())} train / {html.escape(cosy_zip_manifest["dev_count"].__str__())} dev；model-only ckpt {cosy_student_summary["checkpoint_size_mb"]}MB。</td>
        <td>先聽 16-step 品質上限；再做 4/3-step 真蒸餾。</td>
      </tr>
    </tbody>
  </table>

  <h2 id="model-stack">不同模型內容物分析</h2>
  <div class="model-stack">
    <section class="stack-card">
      <h3>Qwen TTS 系列</h3>
      <ul>
        <li><b>VoiceDesign：</b>文字描述控制音色、情緒、韻律。</li>
        <li><b>Base：</b>reference audio clone；需要 ref wav + ref text。</li>
        <li><b>Tokenizer：</b>把 speech 壓成 12Hz 離散 token。</li>
        <li><b>定位：</b>大老師 / 聲音設計 / clone 對照。</li>
      </ul>
    </section>
    <section class="stack-card">
      <h3>CosyVoice2</h3>
      <ul>
        <li><b>Text / LLM：</b>處理文字與語意 token。</li>
        <li><b>Speaker embedding：</b>從 reference 抽聲紋條件。</li>
        <li><b>Speech token / flow：</b>生成 acoustic representation。</li>
        <li><b>定位：</b>目前最強 reference clone teacher。</li>
      </ul>
    </section>
    <section class="stack-card">
      <h3>ZipVoice</h3>
      <ul>
        <li><b>Text encoder：</b>文字 / 拼音條件。</li>
        <li><b>Flow decoder：</b>從 noise 走到 speech latent。</li>
        <li><b>Vocoder：</b>把 latent / mel 轉 waveform。</li>
        <li><b>定位：</b>手機化 student；重點是 few-step + ONNX/int8。</li>
      </ul>
    </section>
  </div>

  <h2 id="qwen-clone">Qwen 可以 clone 嗎？</h2>
  <div class="notice">
    <b>可以，但要換模型。</b>目前報告中的 Qwen 是 <code>1.7B-VoiceDesign-4bit</code>，它適合用文字描述設計聲音；真正 reference clone 要用 <code>Qwen3-TTS-12Hz-1.7B-Base</code> 或 <code>0.6B-Base</code>，輸入 <code>ref_audio</code> + <code>ref_text</code>。
  </div>
  <p>我們已經有本機 <code>0.6B-Base-4bit</code> cache，可以先用它做 smoke test；若效果值得，再下載/跑 <code>1.7B-Base</code> 做正式比較。公平比較方式是餵 Cosy 用過的同一批 reference dataset，產同一批句子，再用同一套 speaker cosine / ASR / 聽感排序。</p>
  <table class="compact-table">
    <thead><tr><th>實驗</th><th>輸入資料</th><th>輸出</th><th>比較方式</th></tr></thead>
    <tbody>
      <tr><td>Qwen Base clone smoke</td><td>同 Cosy 的 clean reference pack + transcript</td><td>3-6 句 clone sample</td><td>先聽有沒有真的像 reference，而不是只像「年輕女生」。</td></tr>
      <tr><td>Qwen Base clone full</td><td>同一批 dataset / 同一批測試句</td><td>完整 audition report</td><td>和 CosyVoice2 並排：聲紋、ASR、速度、穩定度。</td></tr>
      <tr><td>Qwen→ZipVoice distill</td><td>Qwen clone 產生的 corpus</td><td>ZipVoice student</td><td>看 Qwen clone teacher 是否比 Cosy teacher 更適合被蒸餾。</td></tr>
    </tbody>
  </table>

  <h2 id="qwen">模型原理 1：Qwen/Q1 1.7B 怎麼模仿聲音</h2>
  <p>Qwen/Q1 1.7B 在我們這條線上扮演 <b>VoiceDesign teacher</b>：不是拿真人 reference 做 clone，而是用文字 prompt 指定聲音條件，例如「台灣國語、低卷舌、年輕女生、溫柔清亮」。工程上它把文字內容與聲音描述一起 condition 到生成模型中，輸出 waveform 或 acoustic tokens。</p>
  <pre>conditioning = concat(text_tokens, voice_design_tokens)
speech_latents ~ p_theta(speech | text, voice_profile)
waveform = vocoder_or_decoder(speech_latents)</pre>
  <p>這種方式的優點是不用資料就能快速找聲音原型；缺點是聲音不是某個真實 speaker 的穩定 embedding，長文本、不同句型、跨批次時可能漂移。所以 Qwen 分支適合先產生 teacher corpus，再讓 ZipVoice 學一個固定近似聲線。</p>
  {sample_grid(qwen_samples)}

  <h2 id="cosy">模型原理 2：CosyVoice2 怎麼模仿聲音</h2>
  <p>Cosy 這條線是 <b>reference-based zero-shot clone</b>。輸入包含：目標文字、reference wav、reference transcript。Cosy 會從 reference 抽 speaker embedding / prompt speech token，讓 acoustic generator 在同一聲線條件下生成新句子。</p>
  <pre>ref_wav -> speaker_encoder -> speaker_embedding e_spk
ref_wav -> speech_tokenizer -> prompt_speech_tokens z_ref
target_text -> text_encoder / LLM -> semantic_tokens
flow/token2wav: p_theta(wav | semantic_tokens, z_ref, e_spk)</pre>
  <p>Cosy 的重點是 reference 品質：如果 reference 有 BGM、旁白、唱歌、剪裁前綴或逐字稿錯，模型會把那些污染當成 speaker/prosody 條件學進輸出。這也是為什麼目前我們要準備乾淨授權資料做 Cosy speaker fine-tune。</p>
  {sample_grid(cosy_samples)}

  <h2 id="flow">數學背景：Flow Matching / Conditional Flow Matching</h2>
  <p>ZipVoice 與 Cosy acoustic 生成都和 flow-matching 類方法有關。直覺上，模型不是自回歸一個 sample 接一個 sample，而是學一個速度場，把雜訊分布一路推到語音分布。</p>
  <pre>x_0 ~ p_data     # 真實或 teacher 語音 latent / mel
x_1 ~ p_noise    # Gaussian noise
x_t = (1 - t) x_0 + t x_1
u_t = x_1 - x_0
L_CFM = E[ || v_theta(x_t, t, cond) - u_t ||^2 ]</pre>
  <p>推論時從 noise 出發，沿著模型預測的 vector field 積分回 speech latent。<code>num_steps</code> 就是 ODE solver 的離散步數。16-step 等於多修幾次，音色和咬字更穩；4-step/3-step 速度快，但如果沒有針對 few-step 訓練，誤差會累積成電子聲、糊字、漏字。</p>
  <pre>x_{{t-dt}} = x_t - dt * v_theta(x_t, t, cond)
repeat N steps, then vocoder(x_0_hat) -> waveform</pre>

  <h2 id="zip-qwen">ZipVoice 怎麼學 Qwen teacher</h2>
  <p>Qwen 分支的重點是 <b>teacher-student distillation</b>：Qwen 1.7B 先產生同聲線 paired corpus，每筆是 <code>text_i, wav_i^teacher</code>。ZipVoice 不需要知道 Qwen 內部權重，只要學 teacher waveform 對應到文字與 speaker condition 的分布。</p>
  <pre>Dataset_Q = {{(text_i, wav_i^Qwen)}} for i = 1..N
Student cond_i = text_encoder(text_i) + prompt/speaker condition
L_student = L_CFM(mel_or_latent(wav_i^Qwen), cond_i)
Optional:
  L_total = L_CFM + lambda_mel * L1(mel_hat, mel_teacher)
                    + lambda_spk * (1 - cos(spk_hat, spk_teacher))</pre>
  <p>我們做過的 Qwen→ZipVoice 版本包含 8-step 品質候選、6-step 加速、3-step Sherpa runtime。量測上，Qwen teacher peak 約 2.2GB RSS；ZipVoice teacher-ref ONNX int8 peak 約 925-959MB；3-step 已載入後平均約 {speed_best["avg_wall_seconds"]:.2f}s/句。</p>
  {sample_grid(qwen_zip_samples)}

  <h2 id="zip-cosy">ZipVoice 怎麼學 Cosy teacher</h2>
  <p>Cosy 分支是同一個 distillation 概念，但 teacher 換成 CosyVoice2 zero-shot / 未來 Cosy fine-tune 後的固定聲線。現在主要資料是 Cosy golden daily 500：用同一個 clear_best2_7s reference 生成 500 句，再轉成 ZipVoice raw TSV。</p>
  <pre># ZipVoice raw TSV
uniq_id<TAB>text<TAB>wav_path

# 我們的 Cosy corpus
train: {cosy_zip_manifest["train_count"]} utterances
dev:   {cosy_zip_manifest["dev_count"]} utterances
teacher: CosyVoice2-0.5B clear_best2_7s</pre>
  <p>Fine-tune 時從官方 ZipVoice base / distill checkpoint 起跑，更新 student decoder，使它在 Cosy teacher 的語料上最小化 flow loss。這是「聲音蒸餾」，不是把 Cosy 0.5B 參數壓縮成 ZipVoice 權重；學生學到的是 Cosy 輸出聲線的分布近似。</p>
  {sample_grid(cosy_zip16_samples)}

  <h2 id="fewstep">ZipVoice 本身的蒸餾：為什麼 16-step、8-step、4-step 差很多</h2>
  <p>ZipVoice-Distill 的目的，是把原本需要較多 ODE steps 的 flow model，訓練成少步數也能靠近多步數 teacher 的模型。工程上分成兩種：</p>
  <table>
    <thead><tr><th>方法</th><th>做法</th><th>風險</th></tr></thead>
    <tbody>
      <tr><td>硬降 step</td><td>同一個模型，推論時把 <code>num_steps=16</code> 改成 <code>4/3/2</code></td><td>快，但 vector field 沒學過這麼粗的積分，容易電子聲、漏字。</td></tr>
      <tr><td>真正 few-step distillation</td><td>用 16-step teacher trajectory / final output 教 4-step student，使低步數軌跡也接近 teacher</td><td>訓練成本較高，但這才是手機即時生成的正路。</td></tr>
    </tbody>
  </table>
  <pre>Teacher: x_T -> ... -> x_0^teacher  using 16 steps
Student: x_T -> x_0^student          using 4 or 3 steps
L_fewstep = || x_0^student - stopgrad(x_0^teacher) ||^2
          + alpha * || mel(student) - mel(teacher) ||_1
          + beta  * speaker/prosody consistency losses</pre>
  <p>我們早期 Qwen 分支的 loss 記錄：{html.escape(str(losses))}。這只能代表訓練 loss 下降，不代表聽感必然穩；TTS 最後還是要聽 ASR 可懂度、speaker similarity、噪音和自然度。</p>
  {sample_grid(qwen_fewstep_samples)}
  {sample_grid(cosy_fewstep_samples)}

  <h2 id="pipeline">音訊生成 pipeline：從文字到播放</h2>
  <div class="two">
    <section class="card">
      <h3>Teacher server pipeline</h3>
      <pre>text normalization
  -> Qwen VoiceDesign or Cosy reference clone
  -> acoustic tokens / mel / latent
  -> vocoder
  -> wav
  -> teacher corpus</pre>
      <p>這條不追手機大小，追聲音上限與 corpus 乾淨度。</p>
    </section>
    <section class="card">
      <h3>Mobile student pipeline</h3>
      <pre>ASR text
  -> text normalization / pinyin
  -> ZipVoice ONNX int8 acoustic model
  -> Vocos / vocoder int8 or optimized runtime
  -> stream or chunked playback</pre>
      <p>這條追低延遲、低記憶體、可懂度和穩定發音。</p>
    </section>
  </div>

  <h2 id="cosy-sft">Cosy fine-tune 要準備的資料</h2>
  <p>如果要認真用 Cosy fine-tune，資料準備比訓練本身更關鍵。目標不是「更多就好」，而是「乾淨、逐字準、聲線一致」。</p>
  <table>
    <thead><tr><th>項目</th><th>最低可跑</th><th>建議正式量</th><th>檢查標準</th></tr></thead>
    <tbody>
      <tr><td>授權</td><td>明確授權研究/內用</td><td>明確授權商用與衍生模型</td><td>保留來源、範圍、可否公開、可否訓練模型。</td></tr>
      <tr><td>音訊量</td><td>10-30 分鐘可做 speaker SFT smoke</td><td>1-3 小時更穩</td><td>單一 speaker，避免情緒跨度太大。</td></tr>
      <tr><td>切句</td><td>每句 2-10 秒</td><td>3-8 秒最舒服</td><td>不要切到字；前後留 100-250ms room tone。</td></tr>
      <tr><td>音質</td><td>mono 16k/24k wav</td><td>24k wav + loudness normalization</td><td>無 BGM、無唱歌、無旁白、無重疊人聲、無爆音。</td></tr>
      <tr><td>逐字稿</td><td>ASR 初稿 + 人工校</td><td>每句人工逐字校正</td><td>口語詞、停頓、語助詞保留；錯字會訓壞發音。</td></tr>
      <tr><td>metadata</td><td><code>wav.scp / text / utt2spk / spk2utt</code></td><td>再加 train/dev/eval split</td><td>單 speaker 可全用同一 spk id。</td></tr>
      <tr><td>特徵</td><td>speaker embedding、speech token</td><td>打成 parquet 給 recipe</td><td>preprocess 成功後才能穩定訓練。</td></tr>
      <tr><td>固定測試集</td><td>10 句</td><td>30-50 句</td><td>每次 fine-tune 後用同一批句子比較。</td></tr>
    </tbody>
  </table>

  <h2>下一步建議</h2>
  <ol>
    <li><b>Qwen 分支：</b>保留作速度實驗與 baseline，不再把它當唯一聲音目標；重點放在 few-step 乾淨度。</li>
    <li><b>Cosy 分支：</b>先做認真 Cosy fine-tune，拿到更穩的 teacher，再重建 500-3000 句 corpus。</li>
    <li><b>ZipVoice：</b>先用 16-step 聽「學像沒有」，再做 4/3-step true distillation。不要只硬降 step。</li>
    <li><b>手機 app：</b>最後包 ZipVoice ONNX int8 + optimized vocoder + speaker/reference cache，模型載入一次，ASR 後直接生成。</li>
  </ol>

  <p class="muted">Generated at /Users/ader/Documents/App/distillation/taiwan_mandarin_low_r/reports/target_route_deep_technical_v1/index.html</p>
</main>
</body>
</html>
"""
    (OUT / "index.html").write_text(html_doc, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
