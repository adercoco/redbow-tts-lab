#!/usr/bin/env python3
"""Local CosyVoice2 web app for authorized Haibara voice generation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torchaudio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from opencc import OpenCC
except Exception:  # pragma: no cover - optional local dependency
    OpenCC = None


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "conan_authorized_voice_refs_v1"
CLIPS = BASE / "clips_24k"
PACKS = BASE / "reference_packs"
WEB = BASE / "haibara_cosy_web_v1"
AUDIO_OUT = WEB / "generated"
COSY_ROOT = ROOT / "external" / "CosyVoice"
TAIWAN_GOLDEN = (
    ROOT
    / "distillation"
    / "taiwan_mandarin_low_r"
    / "golden_teacher"
    / "cosyvoice2_clear_best2_line04_v1"
)
CHARACTER_PACKS = ROOT / "distillation" / "character_voice_collection_refs_v1" / "reference_packs"
SAMPLE_RATE = 24_000
T2S = OpenCC("t2s") if OpenCC else None
FALLBACK_T2S = str.maketrans(
    {
        "這": "这",
        "個": "个",
        "們": "们",
        "妳": "你",
        "願": "愿",
        "話": "话",
        "會": "会",
        "邊": "边",
        "著": "着",
        "覺": "觉",
        "線": "线",
        "結": "结",
        "論": "论",
        "靜": "静",
        "點": "点",
        "沒": "没",
        "麼": "么",
        "簡": "简",
        "體": "体",
        "聲": "声",
        "輸": "输",
        "入": "入",
        "確": "确",
        "認": "认",
        "擔": "担",
        "幫": "帮",
        "開": "开",
        "藥": "药",
        "測": "测",
        "試": "试",
        "變": "变",
        "機": "机",
        "錄": "录",
        "訊": "讯",
        "傳": "传",
        "轉": "转",
        "產": "产",
        "給": "给",
        "從": "从",
        "網": "网",
        "頁": "页",
        "選": "选",
        "幾": "几",
        "對": "对",
        "錯": "错",
        "慘": "惨",
        "睹": "睹",
        "優": "优",
        "化": "化",
        "畫": "画",
        "面": "面",
        "樣": "样",
        "偵": "侦",
        "探": "探",
        "柯": "柯",
        "南": "南",
        "整": "整",
        "體": "体",
        "希": "希",
        "望": "望",
        "螢": "萤",
        "幕": "幕",
        "氣": "气",
        "時": "时",
        "應": "应",
        "該": "该",
        "現": "现",
        "實": "实",
        "辦": "办",
        "醫": "医",
        "學": "学",
        "發": "发",
        "現": "现",
        "說": "说",
        "讓": "让",
        "聽": "听",
        "過": "过",
        "還": "还",
        "後": "后",
        "裡": "里",
        "嗎": "吗",
        "嗎": "吗",
        "媽": "妈",
        "東": "东",
        "決": "决",
        "兩": "两",
        "個": "个",
        "辦": "办",
        "聲": "声",
        "與": "与",
        "將": "将",
        "來": "来",
        "幹": "干",
        "淨": "净",
        "乾": "干",
        "麼": "么",
        "為": "为",
        "為": "为",
        "無": "无",
        "關": "关",
        "係": "系",
        "緊": "紧",
        "張": "张",
        "氣": "气",
        "實": "实",
        "樣": "样",
        "檢": "检",
        "壞": "坏",
        "懶": "懒",
        "頭": "头",
        "錢": "钱",
        "幫": "帮",
        "覺": "觉",
        "歡": "欢",
        "寶": "宝",
        "貝": "贝",
        "買": "买",
        "賣": "卖",
        "發": "发",
        "票": "票",
        "獎": "奖",
        "點": "点",
        "嚴": "严",
        "飯": "饭",
        "燈": "灯",
        "颱": "台",
        "風": "风",
        "喔": "喔",
        "啦": "啦",
    }
)

WEB.mkdir(parents=True, exist_ok=True)
AUDIO_OUT.mkdir(parents=True, exist_ok=True)

PRESETS = {
    "haibara": {
        "label": "灰原",
        "description": "授權灰原短 reference。對短句比 clean2/best3 穩，也避開「給打開」污染。",
        "audio": CLIPS / "haibara_002_haibara_ep751_000112000_000116500.wav",
        "clips": [],
        "text": "受不了。我把感冒药找出来给你吃吧。",
    },
    "taiwan_soft": {
        "label": "台灣溫柔女聲",
        "description": "你之前選中的 CosyVoice2 golden teacher 風格，不標示為真人聲音。",
        "audio": TAIWAN_GOLDEN / "reference_clear_best2_7s.wav",
        "clips": [],
        "text": "所以我当时就说,我想要做一张疗愈人的专辑。 开始当然就是我们的提案会议,我就提出了因为多年",
    },
    "ryotsu": {
        "label": "兩津勘吉",
        "description": "授權角色片段 best3 reference，CosyVoice2 zero-shot clone。",
        "audio": CHARACTER_PACKS / "ryotsu_best3_24k.wav",
        "clips": [],
        "text": "嗯嗯 真是的 怎么会这样的 诶 什么 这是 到底是什么 请慢啦 这 这 这不是 这是什么 话又说回来，我老了之后恐怕也是单身的。",
    },
    "shinchan": {
        "label": "野原新之助",
        "description": "授權角色片段 kid4 reference，CosyVoice2 zero-shot clone。",
        "audio": CHARACTER_PACKS / "shinchan_kid4_24k.wav",
        "clips": [],
        "text": "脑中喔 走开啦 走掉了啦 喂 是我的那是我的 爸爸 妈妈 我的动感照呢 一定是剧旅行的 不过",
    },
    "misae": {
        "label": "野原美牙",
        "description": "授權角色片段 best3 reference，CosyVoice2 zero-shot clone。",
        "audio": CHARACTER_PACKS / "misae_best3_24k.wav",
        "clips": [],
        "text": "今天到底是吹了什么风啊 你们两个在吵什么 都没什么好的嘛 这个好像太够了",
    },
}


class GenerateRequest(BaseModel):
    text: str
    preset: str = "haibara"


def rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    return 20 * np.log10(rms + 1e-12)


def normalize(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    audio = audio.astype(np.float32)
    audio = audio - float(np.mean(audio))
    gain = 10 ** ((target_db - rms_db(audio)) / 20)
    audio = audio * min(gain, 8.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.96:
        audio = audio * (0.96 / peak)
    return np.clip(audio, -0.98, 0.98).astype(np.float32)


def biquad_filter(audio: np.ndarray, sr: int, kind: str, cutoff: float, q: float = 0.707) -> np.ndarray:
    omega = 2 * np.pi * cutoff / sr
    alpha = np.sin(omega) / (2 * q)
    cosw = np.cos(omega)
    if kind == "lowpass":
        b0 = (1 - cosw) / 2
        b1 = 1 - cosw
        b2 = (1 - cosw) / 2
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha
    elif kind == "highpass":
        b0 = (1 + cosw) / 2
        b1 = -(1 + cosw)
        b2 = (1 + cosw) / 2
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha
    else:
        raise ValueError(f"Unsupported filter kind: {kind}")
    b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    out = np.zeros_like(audio, dtype=np.float32)
    x1 = x2 = y1 = y2 = 0.0
    for index, x0 in enumerate(audio.astype(np.float32)):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[index] = y0
        x2, x1, y2, y1 = x1, x0, y1, y0
    return out


def soft_gate(audio: np.ndarray, sr: int, threshold_db: float = -46.0) -> np.ndarray:
    frame = max(256, int(sr * 0.02))
    hop = max(128, frame // 2)
    envelope = np.ones(audio.size, dtype=np.float32)
    threshold = 10 ** (threshold_db / 20)
    for start in range(0, max(audio.size, 1), hop):
        end = min(audio.size, start + frame)
        if end <= start:
            continue
        rms = float(np.sqrt(np.mean(audio[start:end] ** 2)) + 1e-12)
        gain = 0.35 if rms < threshold else 1.0
        envelope[start:end] = np.minimum(envelope[start:end], gain)
    if envelope.size > 8:
        kernel = np.hanning(min(envelope.size, hop * 2 + 1)).astype(np.float32)
        kernel /= float(np.sum(kernel))
        envelope = np.convolve(envelope, kernel, mode="same").astype(np.float32)
        envelope = np.clip(envelope, 0.35, 1.0)
    return audio * envelope


def postprocess_audio_file(path: Path, preset: str) -> dict[str, float]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    before_rms = rms_db(audio)
    before_peak = 20 * np.log10(float(np.max(np.abs(audio))) + 1e-12) if audio.size else -120.0
    audio = biquad_filter(audio, sr, "highpass", 70.0)
    if preset == "haibara":
        audio = soft_gate(audio, sr, threshold_db=-44.0)
        low = biquad_filter(audio, sr, "lowpass", 6800.0)
        audio = (0.82 * audio + 0.18 * low).astype(np.float32)
        target_db = -20.0
    elif preset == "taiwan_soft":
        bright = audio - biquad_filter(audio, sr, "lowpass", 3600.0)
        audio = audio + 0.16 * bright
        target_db = -19.5
    else:
        target_db = -20.0
    audio = normalize(audio, target_db=target_db)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.89:
        audio = audio * (0.89 / peak)
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_16")
    return {
        "before_rms_db": round(float(before_rms), 2),
        "before_peak_db": round(float(before_peak), 2),
        "after_rms_db": round(float(rms_db(audio)), 2),
        "after_peak_db": round(float(20 * np.log10(float(np.max(np.abs(audio))) + 1e-12)), 2)
        if audio.size
        else -120.0,
    }


def ensure_clean_packs() -> None:
    silence = np.zeros(int(0.22 * SAMPLE_RATE), dtype=np.float32)
    for preset in PRESETS.values():
        audio_path = Path(preset["audio"])
        clips = [Path(path) for path in preset.get("clips", [])]
        if audio_path.exists() or not clips:
            continue
        chunks = []
        for clip in clips:
            audio, _ = sf.read(str(clip), dtype="float32", always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            chunks.extend([audio.astype(np.float32), silence])
        pack = normalize(np.concatenate(chunks), target_db=-20.0)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(audio_path), pack, SAMPLE_RATE, subtype="PCM_16")


ensure_clean_packs()

sys.path.insert(0, str(COSY_ROOT))
sys.path.insert(0, str(COSY_ROOT / "third_party" / "Matcha-TTS"))
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

model = AutoModel(model_dir=str(COSY_ROOT / "pretrained_models" / "CosyVoice2-0.5B"))
lock = threading.Lock()
whisper_model = None
whisper_lock = threading.Lock()
app = FastAPI(title="Red Bow Cosy Voice Changer")


def clean_text(text: str) -> str:
    text = " ".join(text.strip().split())
    if len(text) > 160:
        raise HTTPException(status_code=400, detail="文字先控制在 160 字內，Cosy 比較穩。")
    if not text:
        raise HTTPException(status_code=400, detail="請輸入文字。")
    return text


def clean_asr_text(text: str) -> str:
    text = clean_text(text)
    # Whisper sometimes hallucinates this filler from leading mic noise/silence.
    # Keep the rule in ASR only so typed text can still intentionally start with it.
    while len(text) > 4 and text.startswith("真的"):
        text = text[2:].lstrip("，,。.!！?？ ")
    return clean_text(text)


def text_for_cosy(text: str) -> str:
    text = stabilize_spoken_text(text)
    if T2S:
        return T2S.convert(text)
    return text.translate(FALLBACK_T2S)


def stabilize_spoken_text(text: str) -> str:
    text = " ".join(text.strip().split())
    text = text.replace("測試測試測試", "測試，測試，測試。")
    text = text.replace("测试测试测试", "测试，测试，测试。")
    text = text.replace("變聲器", "變聲器。")
    text = text.replace("变声器", "变声器。")
    text = text.replace("。。", "。")
    if not text.endswith(("。", "！", "？", ".", "!", "?")):
        text += "。"
    return text


def output_path(text: str, preset: str) -> Path:
    digest = hashlib.sha1(f"{preset}\n{text}\n{time.time_ns()}".encode("utf-8")).hexdigest()[:16]
    return AUDIO_OUT / f"redbow_{preset}_{digest}.wav"


def transcribe_audio(path: Path) -> str:
    global whisper_model
    import librosa
    import whisper

    with whisper_lock:
        if whisper_model is None:
            whisper_model = whisper.load_model("small")
        audio, _ = librosa.load(str(path), sr=16_000, mono=True)
        trimmed, _ = librosa.effects.trim(audio, top_db=32)
        if trimmed.size > int(16_000 * 0.2):
            audio = trimmed
        result = whisper_model.transcribe(
            audio.astype(np.float32),
            language="zh",
            task="transcribe",
            fp16=False,
            condition_on_previous_text=False,
            temperature=0,
            no_speech_threshold=0.5,
            logprob_threshold=-1.0,
            initial_prompt="請逐字轉錄繁體中文口語，不要加入沒有聽到的詞。",
        )
    return " ".join(str(result.get("text", "")).strip().split())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    options = "\n".join(
        f'<option value="{key}" {"selected" if key == "haibara" else ""}>{preset["label"]}</option>'
        for key, preset in PRESETS.items()
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>紅色蝴蝶結變聲器</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #201d19;
      --muted: #706b62;
      --line: #ded6c9;
      --red: #bd1f2d;
      --red-mid: #d93442;
      --red-dark: #74121a;
      --gold: #d5a640;
      --gold-dark: #8f6822;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Noto Sans TC", "PingFang TC", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }}
    main {{
      width: min(920px, calc(100vw - 28px));
      height: 100dvh;
      margin: 0 auto;
      padding: 14px 0 max(12px, env(safe-area-inset-bottom));
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .bow-button {{
      display: block;
      order: 9;
      width: min(350px, 92vw);
      height: min(210px, 27dvh);
      min-height: 148px;
      margin: auto auto 0;
      position: relative;
      border: 0;
      background: transparent;
      padding: 0;
      cursor: pointer;
      touch-action: none;
      filter: drop-shadow(0 18px 22px rgba(68, 24, 18, .22));
    }}
    .wing {{
      position: absolute;
      top: 19%;
      width: 44%;
      height: 62%;
      background:
        radial-gradient(circle at 50% 45%, rgba(255,255,255,.22), transparent 34%),
        linear-gradient(145deg, #eb4a55 0%, var(--red-mid) 34%, var(--red) 72%, #8f1720 100%);
      border: 4px solid var(--red-dark);
      box-shadow: inset 0 7px 12px rgba(255,255,255,.22), inset 0 -13px 20px rgba(80,0,0,.23);
    }}
    .wing.left {{
      left: 1%;
      border-radius: 92px 26px 72px 30px;
      transform: skewY(-8deg) rotate(-2deg);
    }}
    .wing.right {{
      right: 1%;
      border-radius: 26px 92px 30px 72px;
      transform: skewY(8deg) rotate(2deg);
    }}
    .fold {{
      content: "";
      position: absolute;
      top: 31%;
      width: 17%;
      height: 43%;
      background: rgba(93, 8, 17, .22);
      filter: blur(.2px);
    }}
    .fold.left {{ left: 29%; transform: skewX(-12deg); border-radius: 60% 30% 40% 60%; }}
    .fold.right {{ right: 29%; transform: skewX(12deg); border-radius: 30% 60% 60% 40%; }}
    .knot {{
      position: absolute;
      left: 37%;
      top: 27%;
      width: 26%;
      height: 50%;
      background: linear-gradient(150deg, #b81f2b, #861720);
      border: 4px solid var(--red-dark);
      border-radius: 18px;
      z-index: 3;
      box-shadow: inset 0 8px 12px rgba(255,255,255,.16), inset 0 -10px 16px rgba(48,0,0,.24);
    }}
    .dial {{
      position: absolute;
      left: 41%;
      top: 36%;
      width: 18%;
      aspect-ratio: 1;
      height: auto;
      z-index: 4;
      border-radius: 50%;
      background:
        radial-gradient(circle at 37% 31%, rgba(255,255,255,.5), transparent 17%),
        radial-gradient(circle, #f0c861 0 23%, #bd8c2f 24% 44%, #7d581d 45% 49%, #e5b94c 50% 100%);
      border: 4px solid var(--gold-dark);
      box-shadow: 0 3px 8px rgba(0,0,0,.25), inset 0 -7px 10px rgba(68,45,11,.3);
    }}
    .dial:before, .dial:after {{
      content: "";
      position: absolute;
      left: 45%;
      top: 10%;
      width: 6%;
      height: 25%;
      border-radius: 2px;
      background: #4e3512;
      transform-origin: 2px 22px;
    }}
    .dial:after {{ transform: rotate(88deg); }}
    .bow-button.recording {{
      transform: translateY(2px) scale(.985);
      filter: drop-shadow(0 10px 14px rgba(120, 18, 24, .32));
    }}
    h1 {{
      order: 1;
      margin: 0;
      text-align: center;
      font-size: clamp(24px, 5vw, 42px);
      line-height: 1.08;
      font-weight: 650;
    }}
    .sub {{
      order: 2;
      text-align: center;
      color: var(--muted);
      margin: 0 auto 4px;
      max-width: 720px;
      line-height: 1.42;
      font-size: 13px;
    }}
    .panel {{
      order: 3;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 0;
    }}
    label {{
      display: block;
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 12px;
    }}
    textarea, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    textarea {{
      min-height: 64px;
      max-height: 88px;
      resize: vertical;
      padding: 10px;
      line-height: 1.55;
      font-size: 16px;
    }}
    select {{
      padding: 9px 10px;
      margin-bottom: 9px;
    }}
    .row {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 9px;
      flex-wrap: wrap;
    }}
    .hint-row {{
      order: 8;
      display: flex;
      justify-content: center;
      color: var(--muted);
      min-height: 22px;
      margin: 0;
      font-size: 13px;
      text-align: center;
    }}
    button {{
      appearance: none;
      border: 1px solid var(--red-dark);
      background: var(--red);
      color: white;
      border-radius: 6px;
      padding: 10px 13px;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }}
    button.secondary {{
      background: #fff;
      color: var(--ink);
      border-color: var(--line);
      font-weight: 500;
    }}
    button:disabled {{ opacity: .55; cursor: wait; }}
    .status {{
      color: var(--muted);
      min-height: 20px;
      font-size: 13px;
    }}
    audio {{
      width: 100%;
      height: 34px;
      margin-top: 8px;
    }}
    .examples {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 7px;
      margin-top: 8px;
    }}
    .examples button {{
      text-align: left;
      background: #fff;
      color: var(--ink);
      border-color: var(--line);
      font-weight: 500;
      min-height: 40px;
      padding: 8px 10px;
      font-size: 13px;
    }}
    @media (max-width: 700px) {{
      main {{ width: min(100vw - 20px, 920px); padding-top: 10px; }}
      .sub {{ display: none; }}
      .examples {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-height: 700px) {{
      h1 {{ font-size: 24px; }}
      textarea {{ min-height: 50px; max-height: 62px; }}
      .examples button {{ min-height: 34px; font-size: 12px; padding: 6px 8px; }}
      .bow-button {{ height: min(170px, 24dvh); min-height: 118px; }}
    }}
  </style>
</head>
<body>
  <main>
    <button class="bow-button" id="record" aria-label="按住錄音">
      <span class="wing left"></span>
      <span class="wing right"></span>
      <span class="fold left"></span>
      <span class="fold right"></span>
      <span class="knot"></span>
      <span class="dial"></span>
    </button>
    <h1>紅色蝴蝶結變聲器</h1>
    <p class="sub">選角色，按住蝴蝶結說話，這台 Mac 會先用 Whisper 聽成文字，再用 CosyVoice2 轉成選到的聲音。也可以直接打字生成。</p>
    <div class="hint-row" id="recordHint">按住蝴蝶結說話，放開後生成</div>
    <section class="panel">
      <label for="preset">角色聲音</label>
      <select id="preset">{options}</select>
      <label for="text">要講的句子</label>
      <textarea id="text">你先冷靜一點，這件事我們慢慢確認就好。</textarea>
      <div class="row">
        <button id="generate">生成聲音</button>
        <button class="secondary" id="clear">清空</button>
        <span class="status" id="status"></span>
      </div>
      <audio id="player" controls preload="none"></audio>
      <div class="examples" id="examples"></div>
    </section>
  </main>
  <script>
    const text = document.getElementById('text');
    const preset = document.getElementById('preset');
    const button = document.getElementById('generate');
    const clear = document.getElementById('clear');
    const status = document.getElementById('status');
    const player = document.getElementById('player');
    const record = document.getElementById('record');
    const recordHint = document.getElementById('recordHint');
    let audioContext = null;
    let processor = null;
    let source = null;
    let stream = null;
    let chunks = [];
    let recording = false;
    let recordingSampleRate = 48000;

    const examples = document.getElementById('examples');
    const examplesByPreset = {{
      haibara: [
        '你先冷靜一點，這件事我們慢慢確認就好。',
        '不用急，線索自己會露出破綻。',
        '我只是覺得，這個說法前後不太一致。',
        '如果你願意的話，我們等一下再一起確認一次。',
        '不要逞強，先把感冒藥吃了再說。',
        '這不是偶然，是有人刻意安排的。'
      ],
      taiwan_soft: [
        '欸你先不要急啦，我們慢慢把事情講清楚。',
        '沒關係，我在這邊聽，你想到哪裡就先說哪裡。',
        '我剛剛想了一下，這樣處理應該會比較穩。',
        '你晚一點到也沒關係，我會先把東西整理好。',
        '如果聲音太硬，就把語氣放輕一點，尾音自然收掉。',
        '我們今天先做到這裡就好，剩下的明天再繼續。'
      ],
      ryotsu: [
        '所長，我這不是偷懶，我是在進行街頭經濟調查啦。',
        '等一下，這個發票搞不好可以中兩百萬，先不要丟！',
        '如果今天中獎，我請大家吃拉麵，當然是你們先墊錢。',
        '我只是借用一下腳踏車，沒想到它自己往派出所外面跑。',
        '這不是貪心，是對財富流動性保持高度敏感。',
        '放心啦，我的計畫只要成功一次，就可以把前面全部補回來。'
      ],
      shinchan: [
        '媽媽，我只是先檢查布丁有沒有壞掉啦。',
        '我沒有偷懶，我是在用屁股思考重要的事情。',
        '漂亮姐姐來了嗎？那我今天可以表現得很成熟喔。',
        '爸爸說要省錢，所以我幫他把零食先吃掉。',
        '我收玩具的速度很快，只是玩具一直不想回家。',
        '等一下啦，我的肚子正在跟我開作戰會議。'
      ],
      misae: [
        '你們兩個先安靜一下，媽媽現在要確認一件事。',
        '小新，你如果再把襪子丟在客廳，我真的會爆炸喔。',
        '廣志，買特價不是亂買，是家庭財務管理。',
        '我只是去超市一下，怎麼回來家裡像被颱風吹過？',
        '今天晚餐很簡單，誰再挑食就自己去洗碗。',
        '等一下，這張折價券明天到期，這件事很嚴重。'
      ]
    }};

    function renderExamples() {{
      const list = examplesByPreset[preset.value] || examplesByPreset.haibara;
      examples.innerHTML = '';
      list.forEach((line) => {{
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'example';
        item.textContent = line;
        item.addEventListener('click', () => {{
          text.value = line;
        }});
        examples.appendChild(item);
      }});
    }}
    preset.addEventListener('change', renderExamples);
    renderExamples();
    clear.addEventListener('click', () => {{
      text.value = '';
      text.focus();
    }});
    async function generateFromText() {{
      const value = text.value.trim();
      if (!value) {{
        status.textContent = '先輸入一句話';
        return;
      }}
      button.disabled = true;
      status.textContent = '生成中...';
      try {{
        const response = await fetch('/api/generate', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ text: value, preset: preset.value }}),
        }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || '生成失敗');
        player.src = data.audio_url + '?t=' + Date.now();
        player.play().catch(() => {{}});
        status.textContent = `完成：${{data.seconds.toFixed(2)}} 秒`;
      }} catch (error) {{
        status.textContent = error.message;
      }} finally {{
        button.disabled = false;
      }}
    }}
    button.addEventListener('click', generateFromText);

    function mergeBuffers(buffers) {{
      const length = buffers.reduce((sum, item) => sum + item.length, 0);
      const result = new Float32Array(length);
      let offset = 0;
      for (const item of buffers) {{
        result.set(item, offset);
        offset += item.length;
      }}
      return result;
    }}

    function encodeWav(samples, sampleRate) {{
      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);
      const writeString = (offset, value) => {{
        for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
      }};
      writeString(0, 'RIFF');
      view.setUint32(4, 36 + samples.length * 2, true);
      writeString(8, 'WAVE');
      writeString(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(36, 'data');
      view.setUint32(40, samples.length * 2, true);
      let offset = 44;
      for (let i = 0; i < samples.length; i++, offset += 2) {{
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }}
      return new Blob([view], {{ type: 'audio/wav' }});
    }}

    async function startRecording(event) {{
      event.preventDefault();
      if (recording) return;
      try {{
        stream = await navigator.mediaDevices.getUserMedia({{ audio: {{ echoCancellation: true, noiseSuppression: true }} }});
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        recordingSampleRate = audioContext.sampleRate;
        chunks = [];
        source = audioContext.createMediaStreamSource(stream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = (evt) => {{
          if (!recording) return;
          chunks.push(new Float32Array(evt.inputBuffer.getChannelData(0)));
        }};
        source.connect(processor);
        processor.connect(audioContext.destination);
        recording = true;
        record.classList.add('recording');
        recordHint.textContent = '錄音中，放開蝴蝶結開始生成';
        status.textContent = '錄音中...';
      }} catch (error) {{
        status.textContent = '麥克風開不起來：' + error.message;
      }}
    }}

    async function stopRecording(event) {{
      event.preventDefault();
      if (!recording) return;
      recording = false;
      record.classList.remove('recording');
      recordHint.textContent = '辨識中...';
      status.textContent = '辨識中...';
      if (processor) processor.disconnect();
      if (source) source.disconnect();
      if (stream) stream.getTracks().forEach((track) => track.stop());
      if (audioContext) await audioContext.close();
      const samples = mergeBuffers(chunks);
      if (samples.length < recordingSampleRate * 0.25) {{
        status.textContent = '錄音太短，再按住說一次';
        recordHint.textContent = '按住蝴蝶結說話，放開後生成';
        return;
      }}
      try {{
        const response = await fetch('/api/transcribe', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'audio/wav' }},
          body: encodeWav(samples, recordingSampleRate),
        }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || '辨識失敗');
        text.value = data.text;
        recordHint.textContent = '聽到：' + data.text;
        await generateFromText();
      }} catch (error) {{
        status.textContent = error.message;
      }}
    }}

    record.addEventListener('pointerdown', startRecording);
    record.addEventListener('pointerup', stopRecording);
    record.addEventListener('pointercancel', stopRecording);
  </script>
</body>
</html>"""


@app.post("/api/transcribe")
async def transcribe(request: Request) -> JSONResponse:
    payload = await request.body()
    if len(payload) < 2048:
        raise HTTPException(status_code=400, detail="錄音太短，請按住蝴蝶結再說一次。")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    started = time.perf_counter()
    try:
        text = transcribe_audio(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
    text = clean_asr_text(text)
    return JSONResponse({"text": text, "seconds": time.perf_counter() - started})


@app.post("/api/generate")
def generate(request: GenerateRequest) -> JSONResponse:
    text = clean_text(request.text)
    cosy_text = text_for_cosy(text)
    if request.preset not in PRESETS:
        raise HTTPException(status_code=400, detail="未知 reference preset。")
    preset = PRESETS[request.preset]
    ref_audio = Path(preset["audio"])
    if not ref_audio.exists():
        raise HTTPException(status_code=500, detail=f"reference 不存在：{ref_audio}")

    output = output_path(text, request.preset)
    started = time.perf_counter()
    audio_metrics = {}
    with lock:
        for index, item in enumerate(
            model.inference_zero_shot(cosy_text, str(preset["text"]), str(ref_audio), stream=False)
        ):
            if index == 0:
                torchaudio.save(str(output), item["tts_speech"], model.sample_rate)
                break
        audio_metrics = postprocess_audio_file(output, request.preset)
    seconds = time.perf_counter() - started
    return JSONResponse(
        {
            "audio_url": f"/audio/{output.name}",
            "seconds": seconds,
            "preset": request.preset,
            "input_text": text,
            "cosy_text": cosy_text,
            "output": str(output),
            "audio_metrics": audio_metrics,
        }
    )


@app.get("/audio/{name}")
def audio(name: str) -> FileResponse:
    if "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="bad filename")
    path = AUDIO_OUT / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/presets")
def presets() -> JSONResponse:
    return JSONResponse(
        {
            key: {
                "label": value["label"],
                "audio": str(value["audio"]),
                "text": value["text"],
            }
            for key, value in PRESETS.items()
        }
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8782"))
    uvicorn.run(app, host=host, port=port, log_level="info")
