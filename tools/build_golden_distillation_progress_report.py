#!/usr/bin/env python3
"""Build a phone-readable progress report for the golden teacher distillation run."""

from __future__ import annotations

import base64
import html
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
CORPUS = BASE / "teacher_cosy_clear_best2_golden_daily_500_v1"
GOLDEN = BASE / "golden_teacher" / "cosyvoice2_clear_best2_line04_v1"
TEXTS = BASE / "distill_texts_golden_daily_v1.jsonl"
DATASET = BASE / "datasets" / "zipvoice_cosy_golden_daily_500_v1"
ZIPVOICE_EGS = ROOT / "external" / "ZipVoice" / "egs" / "zipvoice"
EXP_DIR = ZIPVOICE_EGS / "exp" / "zipvoice_cosy_golden_daily_500_decoder_v1"
TRAIN_LOG = BASE / "logs" / "zipvoice_cosy_golden_daily_500_decoder_train.log"
OUT_DIR = BASE / "reports" / "golden_cosy_zipvoice_distillation_progress_v1"
OUT = OUT_DIR / "golden_cosy_zipvoice_distillation_progress_v1_standalone.html"


def audio_data_uri(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 / 1024


def read_manifest() -> list[dict]:
    path = CORPUS / "manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def read_dataset_summary() -> dict:
    path = DATASET / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_train_status() -> dict:
    status = {
        "state": "not_started",
        "last_line": "",
        "initial_valid_loss": "",
        "latest_checkpoint": "",
        "checkpoint_size_mb": 0.0,
    }
    if TRAIN_LOG.exists():
        status["state"] = "running_or_completed"
        lines = TRAIN_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        interesting = [
            line
            for line in lines
            if "global_batch_idx" in line
            or "checkpoint" in line.lower()
            or "Training started" in line
            or "validation:" in line
        ]
        if interesting:
            status["last_line"] = interesting[-1]
        for line in lines:
            if "global_batch_idx: 0, validation:" in line and "loss=" in line:
                status["initial_valid_loss"] = line.split("loss=", 1)[1].split(",", 1)[0].strip()

    if EXP_DIR.exists():
        checkpoints = sorted(EXP_DIR.glob("checkpoint-*.pt"), key=lambda p: p.stat().st_mtime)
        if checkpoints:
            latest = checkpoints[-1]
            status["latest_checkpoint"] = str(latest.relative_to(ROOT))
            status["checkpoint_size_mb"] = size_mb(latest)
    return status


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def audio_card(title: str, text: str, path: Path, meta: str) -> str:
    if not path.exists():
        return ""
    return f"""
      <section class="audio-card">
        <div class="audio-title">{esc(title)}</div>
        <p>{esc(text)}</p>
        <audio controls preload="metadata" src="{audio_data_uri(path)}"></audio>
        <div class="meta">{esc(meta)}</div>
      </section>
    """


def main() -> int:
    rows = read_manifest()
    ok = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    secs = [row["seconds"] for row in ok if isinstance(row.get("seconds"), (int, float))]
    avg_sec = statistics.mean(secs) if secs else 0
    median_sec = statistics.median(secs) if secs else 0
    total_gen_min = sum(secs) / 60 if secs else 0

    text_count = sum(1 for line in TEXTS.read_text(encoding="utf-8").splitlines() if line.strip()) if TEXTS.exists() else 0
    audio_count = len(list((CORPUS / "audio").glob("*.wav"))) if (CORPUS / "audio").exists() else 0
    raw_count = len(list((CORPUS / "raw_audio").glob("*.wav"))) if (CORPUS / "raw_audio").exists() else 0
    dataset_summary = read_dataset_summary()
    train_status = read_train_status()
    train_count = dataset_summary.get("train_count", 0)
    dev_count = dataset_summary.get("dev_count", 0)
    fbank_train = ZIPVOICE_EGS / "data" / "fbank" / "custom-cosy-golden-daily_cuts_train.jsonl.gz"
    fbank_dev = ZIPVOICE_EGS / "data" / "fbank" / "custom-cosy-golden-daily_cuts_dev.jsonl.gz"
    fbank_ready = fbank_train.exists() and fbank_dev.exists()

    sample_ids = ["golden_daily_0001", "golden_daily_0002", "golden_daily_0008", "golden_daily_0027", "golden_daily_0101", "golden_daily_0201"]
    samples = {row["id"]: row for row in ok if row.get("id") in sample_ids}
    sample_html = [
        audio_card(
            "Golden Teacher Anchor",
            "如果你愿意的话，我们等一下再一起确认一次。",
            GOLDEN / "golden_teacher.wav",
            "User selected CosyVoice2 clear_best2_7s line_04 as golden teacher.",
        )
    ]
    for sample_id in sample_ids:
        row = samples.get(sample_id)
        if not row:
            continue
        seconds = row.get("seconds")
        seconds_note = f"generated in {seconds:.2f}s" if isinstance(seconds, (int, float)) else "recovered from existing wav"
        sample_html.append(
            audio_card(
                f"Cosy Teacher Corpus · {sample_id}",
                row["text"],
                Path(row["audio"]),
                f"CosyVoice2-0.5B · clear_best2_7s · {seconds_note}",
            )
        )

    command_blocks = {
        "0. 背景分批生成": "tools/run_golden_cosy_background_batches.sh\n# 目前用 detached background job 跑，避免 CosyVoice 長輸出把 Codex Desktop 壓到重啟。",
        "1. 生成日常訓練句": ".venv-cosyvoice/bin/python tools/build_golden_daily_distill_texts.py --count 500",
        "2. 生成 Cosy teacher corpus": ".venv-cosyvoice/bin/python tools/generate_cosy_teacher_corpus.py --texts distillation/taiwan_mandarin_low_r/distill_texts_golden_daily_v1.jsonl --limit 500 --pack-id clear_best2_7s --out distillation/taiwan_mandarin_low_r/teacher_cosy_clear_best2_golden_daily_500_v1 --resume",
        "3. 切 ZipVoice train/dev": ".venv-zipvoice/bin/python tools/prepare_cosy_zipvoice_training_data.py --corpus distillation/taiwan_mandarin_low_r/teacher_cosy_clear_best2_golden_daily_500_v1 --out-dir distillation/taiwan_mandarin_low_r/datasets/zipvoice_cosy_golden_daily_500_v1 --dev-count 32 --sync-zipvoice-egs --prefix custom-cosy-golden-daily",
        "4. 轉 ZipVoice manifest/fbank": "PYTHONPATH=../../:$PYTHONPATH .venv-zipvoice/bin/python -m zipvoice.bin.prepare_dataset / prepare_tokens / compute_fbank",
        "5. Fine-tune ZipVoice": "PYTHONPATH=../../:$PYTHONPATH .venv-zipvoice/bin/python -m zipvoice.bin.train_zipvoice --finetune 1 --checkpoint download/zipvoice/model.pt --train-manifest data/fbank/custom-cosy-golden-daily_cuts_train.jsonl.gz --dev-manifest data/fbank/custom-cosy-golden-daily_cuts_dev.jsonl.gz",
        "6. 匯出手機候選": "Export PyTorch checkpoint to ONNX, then dynamic int8 quantization for sherpa-onnx mobile runtime.",
    }
    commands_html = "\n".join(
        f"<h3>{esc(title)}</h3><pre>{esc(cmd)}</pre>" for title, cmd in command_blocks.items()
    )

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Golden Cosy → ZipVoice 蒸餾進度報告</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f3ec;
      --paper: #fffdf8;
      --ink: #25211c;
      --muted: #6c6258;
      --line: #e3d8ca;
      --red: #b43b3b;
      --blue: #255f85;
      --green: #3f6f56;
      --soft: #f2e7db;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      line-height: 1.62;
      font-size: 16px;
    }}
    main {{
      width: min(760px, 100%);
      margin: 0 auto;
      padding: 20px 16px 48px;
    }}
    header {{
      padding: 24px 0 14px;
    }}
    .eyebrow {{
      color: var(--red);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    h1 {{
      font-size: clamp(28px, 8vw, 44px);
      line-height: 1.08;
      margin: 8px 0 12px;
      letter-spacing: 0;
    }}
    .subtitle {{
      font-size: 18px;
      color: var(--muted);
      margin: 0;
    }}
    section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin: 14px 0;
      box-shadow: 0 1px 0 rgba(58, 47, 35, 0.04);
    }}
    h2 {{
      font-size: 22px;
      line-height: 1.2;
      margin: 0 0 10px;
      letter-spacing: 0;
    }}
    h3 {{
      font-size: 17px;
      margin: 18px 0 8px;
    }}
    p {{ margin: 8px 0; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .metric {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .metric b {{
      display: block;
      font-size: 24px;
      line-height: 1.15;
      color: var(--blue);
    }}
    .metric span {{
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-top: 4px;
    }}
    .callout {{
      border-left: 4px solid var(--red);
      background: #fff7f4;
      padding: 12px 14px;
      border-radius: 6px;
      margin: 12px 0;
    }}
    ol, ul {{ padding-left: 22px; }}
    li {{ margin: 8px 0; }}
    code {{
      background: #eee5da;
      padding: 2px 5px;
      border-radius: 5px;
      word-break: break-word;
    }}
    pre {{
      margin: 8px 0 14px;
      padding: 12px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: #211d19;
      color: #fff8ed;
      border-radius: 8px;
      font-size: 13px;
      line-height: 1.5;
    }}
    audio {{
      width: 100%;
      margin-top: 8px;
    }}
    .audio-card {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin: 10px 0;
      box-shadow: none;
    }}
    .audio-title {{
      font-weight: 800;
      color: var(--green);
      margin-bottom: 4px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin-top: 10px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      padding: 9px 6px;
    }}
    th {{ color: var(--muted); font-weight: 700; }}
    .path {{
      font-size: 13px;
      color: var(--muted);
      word-break: break-word;
    }}
    @media (max-width: 520px) {{
      main {{ padding: 16px 12px 40px; }}
      section {{ padding: 15px; }}
      table, thead, tbody, th, td, tr {{ display: block; }}
      thead {{ display: none; }}
      tr {{ border-bottom: 1px solid var(--line); padding: 8px 0; }}
      td {{ border-bottom: 0; padding: 5px 0; }}
      td::before {{
        content: attr(data-label);
        display: block;
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
      }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">2026-06-04 · Taiwan Mandarin Low-R</div>
    <h1>Golden Cosy → ZipVoice 蒸餾進度報告</h1>
    <p class="subtitle">目前重點：先把你選中的 CosyVoice2 golden teacher 固定下來，生成較大的老師語料，再讓 ZipVoice 學這個聲音。</p>
  </header>

  <section>
    <h2>目前狀態</h2>
    <div class="callout">
      這份報告很重要的一點：<b>500 句 teacher corpus、ZipVoice train/dev manifest、fbank 都已經完成</b>，現在正在用官方 ZipVoice base checkpoint 做 fine-tune。新的 ZipVoice student 還在訓練中，尚未到可試聽/匯出 ONNX 的階段。
    </div>
    <div class="grid">
      <div class="metric"><b>{len(ok)} / {text_count}</b><span>Cosy teacher corpus 已生成句數</span></div>
      <div class="metric"><b>{audio_count}</b><span>normalized wav 音檔</span></div>
      <div class="metric"><b>{size_mb(CORPUS):.0f} MB</b><span>目前 corpus 資料夾大小</span></div>
      <div class="metric"><b>{avg_sec:.2f}s</b><span>Cosy 平均生成時間 / 句</span></div>
      <div class="metric"><b>{median_sec:.2f}s</b><span>Cosy median 生成時間 / 句</span></div>
      <div class="metric"><b>{total_gen_min:.1f} 分</b><span>目前累積生成時間</span></div>
      <div class="metric"><b>{train_count} / {dev_count}</b><span>ZipVoice train / dev split</span></div>
      <div class="metric"><b>{"完成" if fbank_ready else "未完成"}</b><span>ZipVoice fbank 特徵</span></div>
      <div class="metric"><b>{train_status["initial_valid_loss"] or "待更新"}</b><span>fine-tune initial dev loss</span></div>
    </div>
    <p class="path">Corpus: {esc(CORPUS.relative_to(ROOT))}</p>
    <p class="path">Dataset: {esc(DATASET.relative_to(ROOT))}</p>
    <p class="path">Fine-tune log: {esc(TRAIN_LOG.relative_to(ROOT))}</p>
  </section>

  <section>
    <h2>這次到底在模仿什麼</h2>
    <p>我們不是讓 ZipVoice 直接模仿一個抽象 prompt，而是先選出一個你實聽後覺得最好的聲音，鎖成 golden teacher。</p>
    <ul>
      <li>Teacher model：<code>CosyVoice2-0.5B</code></li>
      <li>固定 reference pack：<code>clear_best2_7s</code></li>
      <li>Golden anchor 句：<code>如果你愿意的话，我们等一下再一起确认一次。</code></li>
      <li>目標聲音：台灣國語低卷舌、清亮溫柔、自然日常，不要主播腔，不要娃娃音。</li>
    </ul>
  </section>

  <section>
    <h2>ZipVoice Fine-tune 目前狀態</h2>
    <table>
      <thead><tr><th>項目</th><th>目前值</th></tr></thead>
      <tbody>
        <tr><td data-label="項目">Base checkpoint</td><td data-label="目前值"><code>external/ZipVoice/egs/zipvoice/download/zipvoice/model.pt</code></td></tr>
        <tr><td data-label="項目">Student 參數量</td><td data-label="目前值">122,664,804 params</td></tr>
        <tr><td data-label="項目">訓練資料</td><td data-label="目前值">{train_count} train / {dev_count} dev，teacher corpus 約 {size_mb(CORPUS):.0f} MB</td></tr>
        <tr><td data-label="項目">訓練設定</td><td data-label="目前值">fm_decoder only、600 iters、base lr 0.00005、checkpoint every 100 iters、max-duration 10、CPU、fp16 off</td></tr>
        <tr><td data-label="項目">目前 dev loss</td><td data-label="目前值">{esc(train_status["initial_valid_loss"] or "初始 validation 尚未寫入")}</td></tr>
        <tr><td data-label="項目">最新 checkpoint</td><td data-label="目前值">{esc(train_status["latest_checkpoint"] or "尚未產生，第一個會在 iter 250")}</td></tr>
        <tr><td data-label="項目">最後 log</td><td data-label="目前值">{esc(train_status["last_line"] or "尚無訓練 log")}</td></tr>
      </tbody>
    </table>
    <p>這一步是「真正蒸餾」的第一層：用 Cosy golden teacher 產出的 500 句文字/語音對，去 fine-tune ZipVoice student。這版先採用 ZipVoice distill recipe 同方向的 decoder-only 更新，讓聲音/韻律適配能在 Mac CPU 上實際完成。等它產生 checkpoint 後，才會進入 16-step 試聽、8/4-step 速度測試、ONNX/int8 匯出。</p>
  </section>

  <section>
    <h2>試聽樣本</h2>
    <p>這裡放的是目前 teacher corpus 的聲音，不是 ZipVoice 新 student。目的是先確認老師資料本身夠不夠穩。</p>
    {''.join(sample_html)}
  </section>

  <section>
    <h2>蒸餾流程白話版</h2>
    <ol>
      <li><b>選老師聲音。</b> 先從 CosyVoice2 多個 reference / 句子裡聽，選出你認可的 golden teacher。</li>
      <li><b>做老師語料。</b> 用同一個 golden reference 生成 500 句日常對話，每句都有文字和老師 wav。</li>
      <li><b>整理成 ZipVoice 訓練格式。</b> 把每句變成 <code>id / text / wav_path</code>，再轉成 ZipVoice 的 manifest 和 fbank 特徵。</li>
      <li><b>Fine-tune ZipVoice。</b> 從官方 ZipVoice base checkpoint 開始，讓它學這批 Cosy 老師語音的音色、口音、語尾和節奏。</li>
      <li><b>完整複製先看 16-step。</b> 16-step 是品質上限檢查，不追速度，先聽學生能不能像老師。</li>
      <li><b>再降到 8 / 4-step。</b> 這是手機速度候選。若 4-step 聲音壞掉，就需要真正 few-step distillation，而不是只在推論時硬降 step。</li>
      <li><b>匯出 ONNX/int8。</b> 最終手機 app 要放的是 ZipVoice ONNX/int8 + vocoder，不是這 500 句老師 wav。</li>
    </ol>
  </section>

  <section>
    <h2>逐步拆解：實際怎麼做</h2>
    <p>這裡把「模仿」和「蒸餾」拆開講。模仿是先讓老師模型穩定講出同一種聲音；蒸餾是再讓小一點、能部署的 ZipVoice student 學那批老師聲音。</p>

    <h3>1. 選老師聲音：先固定要學的目標</h3>
    <p><b>目的：</b>不要每次都換聲音。TTS 蒸餾最怕老師自己不穩，今天像 A、明天像 B，student 會學成平均臉。</p>
    <p><b>這次做法：</b>先用 CosyVoice2 試不同 reference pack 和不同句子，最後你選中 <code>clear_best2_7s / line_04</code>，句子是 <code>如果你愿意的话，我们等一下再一起确认一次。</code>。這個檔案被鎖成 golden teacher。</p>
    <p><b>輸入：</b>你授權的女聲 reference clip，加上 CosyVoice2 zero-shot cloning。</p>
    <p><b>輸出：</b><code>golden_teacher.wav</code>、<code>reference_clear_best2_7s.wav</code>、<code>manifest.json</code>。之後所有實驗都要回到這個聲音做比較。</p>

    <h3>2. 做老師語料：讓同一個老師講很多句</h3>
    <p><b>目的：</b>ZipVoice 不能只靠一兩句學出穩定聲音。它需要看到同一個 speaker 在不同文字、語氣、長短句裡怎麼講。</p>
    <p><b>這次做法：</b>先生成 500 句偏日常的簡中句子，包含等人、買飲料、安慰、輕聲確認、低卷舌/尾音自然等內容。再用同一個 <code>clear_best2_7s</code> reference 讓 CosyVoice2 逐句生成 wav。</p>
    <p><b>為什麼用簡中：</b>前面實測簡中輸入比較容易讓模型吐出接近台灣國語的普通話口音；繁中在某些模型會比較不穩或變成讀稿味。</p>
    <p><b>每筆資料長這樣：</b></p>
    <pre>id: golden_daily_0001
text: 如果你愿意的话，我们等一下再一起确认一次。
audio: teacher_cosy_clear_best2_golden_daily_500_v1/audio/golden_daily_0001.wav
raw_audio: teacher_cosy_clear_best2_golden_daily_500_v1/raw_audio/golden_daily_0001.wav
teacher_pack_id: clear_best2_7s</pre>
    <p><b>注意：</b>這一步還不是 ZipVoice 新模型，只是建立「老師答案集」。之後 student 要學的就是這些文字對應的老師聲音。</p>

    <h3>3. 整理成 ZipVoice 訓練格式</h3>
    <p><b>目的：</b>Cosy 產出的 manifest 不是 ZipVoice 訓練程式直接吃的格式，所以要轉成 ZipVoice recipe 使用的 TSV / Lhotse manifest / fbank。</p>
    <p><b>第一層 TSV：</b>每一行是 <code>id TAB text TAB wav_path</code>。例如：</p>
    <pre>golden_daily_0001    如果你愿意的话，我们等一下再一起确认一次。    /Users/ader/Documents/App/distillation/.../golden_daily_0001.wav</pre>
    <p><b>Train/dev split：</b>大多數句子放 train，最後一小部分放 dev。dev 不參與學習，只拿來看模型是不是越訓越偏、越訓越壞。</p>
    <p><b>ZipVoice 需要的中間檔：</b></p>
    <ul>
      <li><code>data/raw/custom-cosy-golden-daily_train.tsv</code>：訓練文字與音檔路徑。</li>
      <li><code>data/manifests/*_cuts_raw_train.jsonl.gz</code>：Lhotse cut manifest。</li>
      <li><code>data/manifests/*_cuts_train.jsonl.gz</code>：加好 tokenizer tokens 的 manifest。</li>
      <li><code>data/fbank/*_cuts_train.jsonl.gz</code>：加好聲學特徵路徑的 manifest。</li>
    </ul>
    <p><b>fbank 是什麼：</b>它是把 wav 轉成模型比較好學的頻譜特徵。訓練時模型不是直接看波形，而是看類似語音頻譜的 representation。</p>

    <h3>4. Fine-tune ZipVoice：讓 student 學老師聲音</h3>
    <p><b>目的：</b>從官方 ZipVoice base checkpoint 出發，不從零訓練。base model 已經會中文 TTS、flow matching、聲音生成；fine-tune 是把它往 golden teacher 的音色和語氣推。</p>
    <p><b>訓練時模型看到什麼：</b>文字 tokens + 老師 wav 的聲學特徵。它學的是「給這段文字時，應該生成接近老師聲音的語音特徵」。</p>
    <p><b>會學到什麼：</b>音色、常見語速、語尾、低卷舌傾向、溫柔清亮的發聲位置。</p>
    <p><b>不一定能完全學到什麼：</b>如果老師語料本身有偶發怪字、情緒飄、不同句子像不同人，student 也會一起學到。所以 teacher corpus 之後要抽聽、必要時刪壞樣本。</p>
    <p><b>這一步產物：</b>ZipVoice checkpoint，例如 <code>checkpoint-xxx.pt</code>。這才是第一個真正的 student model。</p>

    <h3>5. 16-step：先看完整複製上限</h3>
    <p><b>step 是什麼：</b>ZipVoice 是 flow-matching 類模型，生成聲音時會從噪聲逐步走向語音。step 越多，修正次數越多，通常品質越好但越慢。</p>
    <p><b>為什麼先看 16-step：</b>這是品質檢查，不是速度檢查。若 16-step 都不像 golden teacher，代表 fine-tune 沒學好，直接看 4-step 沒意義。</p>
    <p><b>我們要聽什麼：</b>像不像同一個女生、尾音是否自然、卷舌是否降低、字有沒有跑掉、短句是否穩、長句是否喘或斷。</p>

    <h3>6. 8 / 4-step：看手機速度候選</h3>
    <p><b>目的：</b>手機上最重要的是一句話多久生成。降低 step 可以少跑幾輪 decoder，所以可能明顯加速。</p>
    <p><b>風險：</b>如果只是推論時硬把 16-step 模型改成 4-step，聲音可能變糊、咬字跑掉、韻律崩掉。這就是你之前聽到 4-step 不如舊版的原因之一。</p>
    <p><b>真正 few-step distillation：</b>不是只改推論參數，而是訓一個本來就被教會「4 step 也要像 16 step / teacher」的 student。這會另外做 teacher trajectory 或 audio-level 對齊，目標是低 step 也保住聲音。</p>

    <h3>7. 匯出 ONNX / int8：變成手機可以放的形狀</h3>
    <p><b>目的：</b>PyTorch checkpoint 適合研究，不適合直接塞 iPhone / Android app。手機部署要轉成 ONNX，再用 sherpa-onnx 或手機端 runtime 載入。</p>
    <p><b>int8 是什麼：</b>把一部分模型權重從 fp32 壓到 8-bit，模型檔變小，記憶體通常也比較低。它不等於一定更快，但通常比較接近手機部署需求。</p>
    <p><b>app 最終會放什麼：</b><code>text_encoder_int8.onnx</code>、<code>fm_decoder_int8.onnx</code>、<code>vocos_24khz.onnx</code>，再加 tokenizer / config。500 句老師 wav 不會塞進 app。</p>
    <p><b>最後驗收：</b>報告要列 model size、peak RSS、模型載入後單句生成時間、RTF、音檔試聽。手機實測比 Mac Python 數字更重要。</p>
  </section>

  <section>
    <h2>蒸餾不是什麼</h2>
    <p>目前的 corpus 生成不是最終產品，它只是做老師資料。真正 student 會在 ZipVoice fine-tune 後出現。</p>
    <table>
      <thead><tr><th>階段</th><th>是不是蒸餾完成</th><th>作用</th></tr></thead>
      <tbody>
        <tr><td data-label="階段">Cosy 生成 teacher corpus</td><td data-label="是不是蒸餾完成">已完成老師資料，不是 student</td><td data-label="作用">建立 ZipVoice 要學的聲音範本。</td></tr>
        <tr><td data-label="階段">ZipVoice fine-tune</td><td data-label="是不是蒸餾完成">正在進行，這是第一層完整蒸餾</td><td data-label="作用">讓 ZipVoice student 學 golden teacher 的聲音。</td></tr>
        <tr><td data-label="階段">ONNX/int8</td><td data-label="是不是蒸餾完成">是部署壓縮</td><td data-label="作用">讓手機可以跑，模型變小、載入方式更接近 app。</td></tr>
        <tr><td data-label="階段">4-step / few-step</td><td data-label="是不是蒸餾完成">是速度蒸餾</td><td data-label="作用">把生成步數降下來，追求接近即時。</td></tr>
      </tbody>
    </table>
  </section>

  <section>
    <h2>實際命令與檔案</h2>
    {commands_html}
  </section>

  <section>
    <h2>接下來怎麼做</h2>
    <ol>
      <li>等待 ZipVoice fine-tune 產生 <code>checkpoint-250.pt</code>、<code>checkpoint-500.pt</code>、<code>checkpoint-750.pt</code>、<code>checkpoint-1000.pt</code>。</li>
      <li>先用 16-step 產出學生試聽，確認完整複製上限。</li>
      <li>再用 8-step / 4-step 測速度與咬字，判斷是否需要真正 few-step distillation。</li>
      <li>產出四組試聽：Golden Cosy teacher、ZipVoice 16-step、ZipVoice 8-step、ZipVoice 4-step。</li>
      <li>報告列模型大小、峰值記憶體、每句生成時間、RTF、音檔試聽。</li>
    </ol>
  </section>

  <section>
    <h2>目前卡住原因</h2>
    <p>CosyVoice2 逐句生成和 ZipVoice CPU fine-tune 都很吃資源，一次把長時間輸出綁在 Codex 互動回合時，Codex Desktop 的工具連線容易中斷，看起來像當機。</p>
    <p>現在改成 detached background job：Cosy corpus 已經補滿 500 句；ZipVoice fine-tune 也用 <code>tools/run_zipvoice_golden_daily_finetune.sh</code> 在背景跑，log 寫到檔案。</p>
    <p>訓練腳本已關掉每個 epoch 都存 1.8GB checkpoint 的行為，只保留每 250 iter 的 checkpoint，避免小資料集短 epoch 造成硬碟爆量。</p>
  </section>
</main>
</body>
</html>
"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_text, encoding="utf-8")
    print(OUT)
    print(f"ok={len(ok)} audio={audio_count} raw={raw_count} size_mb={size_mb(CORPUS):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
