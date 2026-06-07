#!/usr/bin/env python3
"""Build original zero-shot clone audition v4 without ZipVoice step-down variants."""

from __future__ import annotations

import base64
import collections
import difflib
import html
import json
import mimetypes
import os
import re
import statistics
from pathlib import Path

import librosa
import onnxruntime
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
V1 = BASE / "clone_audition_v1"
V2 = BASE / "clone_audition_v2_large_models"
V3 = BASE / "clone_audition_v3_dataset_models"
OUT = BASE / "clone_audition_v4_original_models"
RESULTS = OUT / "all_results_asr_scored.json"
REPORT = OUT / "index.html"
STANDALONE = OUT / "clone_audition_v4_original_models_standalone.html"
CAMPLUS = ROOT / "external" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B" / "campplus.onnx"

MODEL_ORDER = [
    "CosyVoice2-0.5B pack",
    "F5-TTS v1 Base pack",
    "GPT-SoVITS v2 single ref",
    "GPT-SoVITS v2 aux refs",
    "IndexTTS2 pack",
    "Qwen3 1.7B VoiceDesign ref attempt",
    "Sherpa ZipVoice int8 zh-min 16-step",
    "Sherpa PocketTTS int8 5-step",
]

TEXT_LABELS = {
    "tw_soft": "先別急",
    "tw_calm": "慢慢說",
    "tw_decide": "慢慢決定",
    "tw_confirm": "一起確認",
}

MODEL_META = {
    "CosyVoice2-0.5B pack": {
        "size_paths": [ROOT / "external" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B"],
        "note": "CosyVoice2 原始 zero-shot clone，固定用 pack_best2_7s reference。",
    },
    "F5-TTS v1 Base pack": {
        "size_label": "約 1.3GB HF/model cache",
        "note": "F5-TTS v1 Base 原始 zero-shot clone，沒有 fine-tune。",
    },
    "GPT-SoVITS v2 single ref": {
        "size_paths": [ROOT / "external" / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"],
        "note": "GPT-SoVITS v2 官方底模，單一 3.27s reference；和 aux refs 分開看。",
    },
    "GPT-SoVITS v2 aux refs": {
        "size_paths": [ROOT / "external" / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"],
        "note": "GPT-SoVITS v2 官方底模，多段 auxiliary refs；上輪有 prompt leakage，這輪仍保留讓你聽。",
    },
    "IndexTTS2 pack": {
        "size_paths": [ROOT / "external" / "index-tts" / "checkpoints"],
        "note": "IndexTTS2 原始 clone，大模型聽感候選；授權需另外確認。",
    },
    "Qwen3 1.7B VoiceDesign ref attempt": {
        "size_paths": [Path.home() / ".cache" / "huggingface" / "hub" / "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"],
        "note": "Qwen3 VoiceDesign ref attempt；CLI 顯示仍走內建 voice，不能視為純 clone。",
    },
    "Sherpa ZipVoice int8 zh-min 16-step": {
        "size_paths": [
            ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min",
            ROOT / "models" / "sherpa" / "vocos_24khz.onnx",
        ],
        "note": "ZipVoice 原始品質聽感版：16-step，不放 4-step/3-step/qint8 vocoder 降階測試。",
    },
    "Sherpa PocketTTS int8 5-step": {
        "size_paths": [ROOT / "models" / "sherpa" / "sherpa-onnx-pocket-tts-int8-2026-01-26"],
        "note": "PocketTTS 原始 clone 小模型；本地 pack README 標示 non-commercial，且前輪內容明顯跑掉。",
    },
}

PENDING = [
    "Chatterbox Turbo: package installed and downloaded about 2.1GB cache, but first MPS smoke generation exceeded the time budget; kept out of this report until a standalone debug run succeeds.",
    "Fish Speech / Fish Audio: not installed locally yet; likely teacher/server class first, not phone runtime.",
    "OpenVoice V2: not installed locally yet; more like tone-color conversion pipeline than direct TTS clone.",
]


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def normalize_zh(text: str) -> str:
    text = text.lower()
    replacements = {"別": "别", "欸": "诶", "緊": "紧", "願": "愿", "話": "话", "線": "线", "覺": "觉", "這": "这", "們": "们"}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))


def asr_similarity(expected: str, actual: str) -> float:
    a = normalize_zh(expected)
    b = normalize_zh(actual)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def repetition_junk(text: str) -> bool:
    compact = normalize_zh(text)
    return len(compact) < 4 or (len(compact) >= 12 and len(set(compact)) / len(compact) < 0.22)


def pass_asr(expected: str, actual: str) -> bool:
    compact = normalize_zh(actual)
    expected_len = max(1, len(normalize_zh(expected)))
    if repetition_junk(actual) or len(compact) > expected_len * 3:
        return False
    return asr_similarity(expected, actual) >= 0.45


def bytes_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def fmt_bytes(value: int | None) -> str:
    if not value:
        return "-"
    size = float(value)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return "-"


def model_size_label(family: str) -> str:
    meta = MODEL_META[family]
    if "size_label" in meta:
        return str(meta["size_label"])
    return fmt_bytes(sum(bytes_of(Path(path)) for path in meta.get("size_paths", [])))


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def load_rows() -> list[dict]:
    rows: list[dict] = []
    v2_rows = load_json(V2 / "all_results_scored.json")
    for row in v2_rows:
        if row.get("family") in {"CosyVoice2-0.5B pack", "F5-TTS v1 Base pack", "IndexTTS2 pack"} and row.get("pack_id") == "pack_best2_7s":
            rows.append(dict(row))
        elif row.get("family") == "GPT-SoVITS v2 aux refs":
            rows.append(dict(row))

    v1_rows = load_json(V1 / "clone_audition_results_scored.json")
    for row in v1_rows:
        if row.get("family") == "GPT-SoVITS v2" and row.get("ref_id") == "ref_01":
            item = dict(row)
            item["family"] = "GPT-SoVITS v2 single ref"
            rows.append(item)
        if row.get("family") == "Qwen3 1.7B VoiceDesign ref attempt" and row.get("ref_id") == "ref_01":
            rows.append(dict(row))

    for row in load_json(OUT / "gpt_sovits_single_extra_results.json"):
        item = dict(row)
        item["family"] = "GPT-SoVITS v2 single ref"
        rows.append(item)
    rows.extend(load_json(OUT / "qwen_ref01_extra_results.json"))
    rows.extend(load_json(OUT / "zipvoice_16step_results.json"))

    v3_rows = load_json(V3 / "sherpa_mobile_results.json")
    for row in v3_rows:
        if row.get("family") == "Sherpa PocketTTS int8 5-step" and row.get("pack_id") == "pack_best2_7s":
            rows.append(dict(row))

    dedup: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.get("status") != "ok" or not row.get("output"):
            continue
        key = (row["family"], row["text_id"])
        if row["text_id"] in TEXT_LABELS and key not in dedup:
            dedup[key] = row
    return [dedup[(family, text_id)] for family in MODEL_ORDER for text_id in TEXT_LABELS if (family, text_id) in dedup]


def load_audio(path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    return audio


def embed(session: onnxruntime.InferenceSession, path: Path) -> torch.Tensor:
    audio = load_audio(path)
    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = session.run(None, {session.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
    vector = torch.tensor(embedding, dtype=torch.float32)
    return torch.nn.functional.normalize(vector, dim=0)


def score_speaker(rows: list[dict]) -> None:
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(CAMPLUS), sess_options=option, providers=["CPUExecutionProvider"])
    cache: dict[str, torch.Tensor] = {}

    def get(path_value: str) -> torch.Tensor | None:
        path = Path(path_value)
        if not path.exists():
            return None
        key = str(path)
        if key not in cache:
            cache[key] = embed(session, path)
        return cache[key]

    for row in rows:
        out = get(str(row.get("output", "")))
        ref_values = [str(row.get("ref_audio", ""))]
        ref_values.extend(str(path) for path in row.get("aux_ref_audio_paths", []))
        refs = [ref for ref in (get(path) for path in ref_values) if ref is not None]
        if out is None or not refs:
            row["speaker_cosine"] = None
            continue
        scores = [torch.dot(ref, out).item() for ref in refs]
        row["speaker_cosine"] = round(max(scores), 4)
        row["speaker_score_note"] = "campplus_cosine_max_over_reference_pool"


def transcribe_rows(rows: list[dict]) -> None:
    model = whisper.load_model("tiny")
    for index, row in enumerate(rows, 1):
        path = Path(row["output"])
        info = sf.info(str(path))
        row["audio_duration_sec"] = round(float(info.duration), 3)
        row["audio_file_size"] = path.stat().st_size
        y, sr = sf.read(str(path), always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=16000)
        result = model.transcribe(y.astype("float32"), language="zh", fp16=False, verbose=False, temperature=0.0, condition_on_previous_text=False)
        text = result.get("text", "").strip()
        row["asr_text"] = text
        row["asr_similarity"] = round(asr_similarity(row.get("text", ""), text), 4)
        row["asr_pass"] = pass_asr(row.get("text", ""), text)
        print(f"[{index:02d}/{len(rows)}] {row['family']} {row['text_id']} pass={row['asr_pass']} sim={row['asr_similarity']:.3f} {text}")


def fmt_sec(value: object) -> str:
    try:
        return f"{float(value):.2f}s"
    except Exception:
        return "-"


def fmt_score(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "-"


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def audio_src(path_value: str, *, standalone: bool) -> str:
    path = Path(path_value)
    if not path.exists():
        return ""
    return data_uri(path) if standalone else esc(os.path.relpath(path, OUT))


def audio_tag(path_value: str, *, standalone: bool) -> str:
    src = audio_src(path_value, standalone=standalone)
    return f'<audio controls preload="none" src="{src}"></audio>' if src else '<div class="missing">沒有音檔</div>'


def badge(text: str, kind: str = "") -> str:
    return f'<span class="badge {esc(kind)}">{esc(text)}</span>'


def stats_by_model(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for family in MODEL_ORDER:
        items = [row for row in rows if row["family"] == family]
        if not items:
            continue
        secs = [float(row.get("seconds") or 0) for row in items]
        scores = [float(row["speaker_cosine"]) for row in items if row.get("speaker_cosine") is not None]
        out[family] = {
            "n": len(items),
            "asr": sum(1 for row in items if row.get("asr_pass")),
            "avg_sec": statistics.mean(secs) if secs else None,
            "avg_score": statistics.mean(scores) if scores else None,
            "size": model_size_label(family),
        }
    return out


def render_cards(rows: list[dict]) -> str:
    stats = stats_by_model(rows)
    pieces = []
    for family in MODEL_ORDER:
        if family not in stats:
            continue
        item = stats[family]
        ok = item["asr"] == item["n"]
        pieces.append(
            f"""
            <article class="model-card">
              <div class="model-top"><h3>{esc(family)}</h3>{badge('ASR ' + str(item['asr']) + '/' + str(item['n']), 'ok' if ok else 'warn')}</div>
              <div class="grid">
                <div><span>模型大小</span><b>{esc(item['size'])}</b></div>
                <div><span>平均生成</span><b>{fmt_sec(item['avg_sec'])}</b></div>
                <div><span>聲紋分數</span><b>{fmt_score(item['avg_score'])}</b></div>
                <div><span>句數</span><b>{item['n']}</b></div>
              </div>
              <p>{esc(MODEL_META[family]['note'])}</p>
            </article>
            """
        )
    return "".join(pieces)


def render_sample(row: dict, *, standalone: bool) -> str:
    return f"""
    <article class="sample">
      <div class="sample-head"><b>{esc(row['family'])}</b><span>{fmt_sec(row.get('seconds'))} · sim {fmt_score(row.get('speaker_cosine'))}</span></div>
      <div class="badges">{badge('ASR ok' if row.get('asr_pass') else 'ASR fail', 'ok' if row.get('asr_pass') else 'warn')}{badge('ASR ' + fmt_score(row.get('asr_similarity')), '')}</div>
      {audio_tag(row.get('output', ''), standalone=standalone)}
      <p class="asr"><b>ASR</b> {esc(row.get('asr_text', ''))}</p>
    </article>
    """


def render_samples(rows: list[dict], *, standalone: bool) -> str:
    by_text: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_text[row["text_id"]].append(row)
    sections = []
    for text_id, label in TEXT_LABELS.items():
        items = sorted(by_text[text_id], key=lambda row: MODEL_ORDER.index(row["family"]))
        if not items:
            continue
        sections.append(
            f"""
            <section class="section">
              <h2>{esc(label)}</h2>
              <p class="expected">{esc(items[0]['text'])}</p>
              <div class="sample-list">{''.join(render_sample(row, standalone=standalone) for row in items)}</div>
            </section>
            """
        )
    return "".join(sections)


def render_references(rows: list[dict], *, standalone: bool) -> str:
    refs: dict[str, dict] = {}
    for row in rows:
        refs.setdefault(row["ref_id"], row)
    parts = []
    for ref_id, row in refs.items():
        parts.append(
            f"""
            <article class="sample ref">
              <div class="sample-head"><b>{esc(ref_id)}</b><span>{esc(row.get('ref_file') or ', '.join(row.get('ref_files', [])))}</span></div>
              <p>{esc(row.get('ref_text'))}</p>
              {audio_tag(row.get('ref_audio', ''), standalone=standalone)}
            </article>
            """
        )
    return "".join(parts)


def render_html(rows: list[dict], *, standalone: bool) -> str:
    pending = "".join(f"<li>{esc(item)}</li>" for item in PENDING)
    mode = "standalone：音檔已嵌入，可傳手機/雲端" if standalone else "local：音檔用相對路徑載入"
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>原始 Clone 模型試聽 v4</title>
  <style>
    :root {{ --bg:#f6f4ef; --paper:#fffdf8; --ink:#211f1c; --muted:#776f65; --line:#ddd4c6; --accent:#b84040; --ok:#226950; --warn:#9c5a00; --soft:#eee7dc; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif; line-height:1.58; }}
    main {{ width:min(980px,100%); margin:0 auto; padding:22px 14px 44px; }}
    header {{ padding:12px 2px 18px; border-bottom:1px solid var(--line); margin-bottom:18px; }}
    .eyebrow {{ color:var(--accent); font-size:13px; font-weight:700; }}
    h1 {{ font-size:clamp(28px,6.7vw,46px); line-height:1.08; margin:7px 0 10px; letter-spacing:0; }}
    h2 {{ font-size:22px; line-height:1.25; margin:0 0 10px; }}
    h3 {{ font-size:17px; line-height:1.3; margin:0; }}
    p {{ margin:8px 0; }}
    .small,.asr,.expected,li {{ color:var(--muted); font-size:14px; }}
    .expected {{ color:var(--ink); background:var(--soft); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
    .callout,.model-card,.section,.sample {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:13px; }}
    .callout {{ margin:14px 0; background:#fff3e0; border-color:#e5cfa8; }}
    .models,.sample-list {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px; }}
    .models {{ margin:16px 0 20px; }}
    .section {{ margin:15px 0; }}
    .model-top,.sample-head {{ display:flex; gap:8px; align-items:flex-start; justify-content:space-between; }}
    .sample-head span {{ color:var(--muted); font-size:12px; text-align:right; min-width:96px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:11px 0; }}
    .grid div {{ background:#f7f1e8; border-radius:8px; padding:8px; }}
    .grid span {{ display:block; color:var(--muted); font-size:12px; }}
    .grid b {{ font-size:15px; overflow-wrap:anywhere; }}
    audio {{ width:100%; margin-top:8px; }}
    .badge {{ display:inline-flex; align-items:center; border-radius:999px; border:1px solid var(--line); padding:3px 8px; font-size:12px; color:var(--muted); background:#fff; white-space:nowrap; }}
    .badge.ok {{ color:var(--ok); border-color:#9ccdbd; background:#eef9f5; }}
    .badge.warn {{ color:var(--warn); border-color:#e2bd75; background:#fff8e8; }}
    .badges {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }}
    @media (max-width:620px) {{ main {{ padding-inline:12px; }} .models,.sample-list {{ grid-template-columns:1fr; }} .model-top,.sample-head {{ display:block; }} .sample-head span {{ display:block; text-align:left; margin-top:4px; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Original clone audition v4 · {esc(mode)}</div>
    <h1>原始 Clone 模型試聽</h1>
    <p>這版先不放 ZipVoice 4-step / 3-step / qint8 vocoder 速度測試。每個候選最多 4 句日常台詞，重點是原始 clone 聽感。</p>
  </header>
  <section class="callout"><b>看法：</b>先聽 Cosy / F5 / IndexTTS2 / ZipVoice 16-step 的自然度，再用 ASR 欄位抓內容是否跑掉。GPT aux 和 PocketTTS 目前保留讓你聽失敗型態。</section>
  <section><h2>模型總表</h2><div class="models">{render_cards(rows)}</div></section>
  <section class="section"><h2>Reference</h2><p class="small">同一批授權 dataset 裁切出的 reference；不是模型輸出。</p><div class="sample-list">{render_references(rows, standalone=standalone)}</div></section>
  {render_samples(rows, standalone=standalone)}
  <section class="section"><h2>本輪未納入</h2><ul>{pending}</ul><p class="small">F5 fine-tuned 40 updates 也先不放主表，因為它不是原始 zero-shot clone。</p></section>
</main>
</body>
</html>"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    print(f"rows={len(rows)}")
    score_speaker(rows)
    transcribe_rows(rows)
    RESULTS.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(render_html(rows, standalone=False), encoding="utf-8")
    STANDALONE.write_text(render_html(rows, standalone=True), encoding="utf-8")
    print(RESULTS)
    print(REPORT)
    print(STANDALONE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
