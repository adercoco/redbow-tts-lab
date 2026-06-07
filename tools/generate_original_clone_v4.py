#!/usr/bin/env python3
"""Generate original zero-shot clone auditions for the Downloads female dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
OUT = BASE / "clone_audition_v4_original_models"
AUDIO = OUT / "audio"
PACKS = BASE / "reference_packs_v1" / "manifest.json"
SINGLE_REF_AUDIO = BASE / "clean_strict_24k" / "female_seg_003_0103.62_0106.89.wav"
SINGLE_REF_TEXT = "所以我当时就说,我想要做一张疗愈人的专辑。"

TESTS = [
    ("line_01", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("line_02", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
    ("line_03", "我觉得这件事情可以慢慢来，不需要马上决定。"),
    ("line_04", "如果你愿意的话，我们等一下再一起确认一次。"),
    ("line_05", "今天下班以后要不要先吃点东西，再回家休息一下。"),
    ("line_06", "这个我大概懂你的意思，我们可以换个比较简单的做法。"),
]


def pack_best2() -> dict:
    packs = json.loads(PACKS.read_text(encoding="utf-8"))
    return next(pack for pack in packs if pack["pack_id"] == "pack_best2_7s")


def write_rows(model_key: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model_key}_results.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


def base_row(family: str, model_key: str, text_id: str, text: str, output: Path, *, ref_mode: str) -> dict:
    pack = pack_best2()
    if ref_mode == "single":
        ref_audio = str(SINGLE_REF_AUDIO)
        ref_text = SINGLE_REF_TEXT
        ref_id = "single_ref_3s"
        ref_files = [SINGLE_REF_AUDIO.name]
    else:
        ref_audio = pack["pack_audio"]
        ref_text = pack["ref_text"]
        ref_id = pack["pack_id"]
        ref_files = pack.get("ref_files", [])
    return {
        "family": family,
        "candidate_id": model_key,
        "model_key": model_key,
        "ref_id": ref_id,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "ref_files": ref_files,
        "text_id": text_id,
        "text": text,
        "output": str(output),
        "clone_mode": "original_zero_shot_clone",
    }


def run_cosy() -> list[dict]:
    import torchaudio

    cosy_root = ROOT / "external" / "CosyVoice"
    sys.path.insert(0, str(cosy_root))
    sys.path.insert(0, str(cosy_root / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    pack = pack_best2()
    model = AutoModel(model_dir=str(cosy_root / "pretrained_models" / "CosyVoice2-0.5B"))
    rows = []
    for text_id, text in TESTS:
        output = AUDIO / "cosyvoice2_0p5b_pack_best2" / f"{text_id}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status = "ok"
        error = ""
        if not output.exists():
            try:
                for index, item in enumerate(
                    model.inference_zero_shot(text, pack["ref_text"], pack["pack_audio"], stream=False)
                ):
                    if index == 0:
                        torchaudio.save(str(output), item["tts_speech"], model.sample_rate)
            except Exception as exc:
                status = "failed"
                error = str(exc)
        row = base_row("CosyVoice2-0.5B original", "cosyvoice2_0p5b_pack_best2", text_id, text, output, ref_mode="pack")
        row.update({"seconds": time.perf_counter() - started, "status": status, "error": error})
        if status != "ok" or not output.exists():
            row["output"] = ""
        rows.append(row)
        print(f"CosyVoice2 {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def run_f5() -> list[dict]:
    pack = pack_best2()
    cli = ROOT / ".venv-f5" / "bin" / "f5-tts_infer-cli"
    rows = []
    for text_id, text in TESTS:
        output = AUDIO / "f5_tts_v1_base_pack_best2" / f"{text_id}.wav"
        work = output.parent / f"_work_{text_id}"
        output.parent.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status = "ok"
        error = ""
        if not output.exists():
            try:
                subprocess.run(
                    [
                        str(cli),
                        "--model",
                        "F5TTS_v1_Base",
                        "--ref_audio",
                        pack["pack_audio"],
                        "--ref_text",
                        pack["ref_text"],
                        "--gen_text",
                        text,
                        "--output_dir",
                        str(work),
                        "--output_file",
                        output.name,
                        "--remove_silence",
                        "--nfe_step",
                        "32",
                    ],
                    cwd=work,
                    check=True,
                )
                generated = work / output.name
                if not generated.exists():
                    raise RuntimeError(f"F5 generated no wav at {generated}")
                shutil.copy2(generated, output)
            except Exception as exc:
                status = "failed"
                error = str(exc)
        row = base_row("F5-TTS v1 Base original", "f5_tts_v1_base_pack_best2", text_id, text, output, ref_mode="pack")
        row.update({"seconds": time.perf_counter() - started, "status": status, "error": error})
        if status != "ok" or not output.exists():
            row["output"] = ""
        rows.append(row)
        print(f"F5 {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def run_index() -> list[dict]:
    index_root = ROOT / "external" / "index-tts"
    os.chdir(index_root)
    sys.path.insert(0, str(index_root))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from indextts.infer_v2 import IndexTTS2

    pack = pack_best2()
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=False,
        device="mps",
        use_cuda_kernel=False,
        use_deepspeed=False,
    )
    rows = []
    for text_id, text in TESTS:
        output = AUDIO / "indextts2_pack_best2" / f"{text_id}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status = "ok"
        error = ""
        if not output.exists():
            try:
                tts.infer(
                    spk_audio_prompt=pack["pack_audio"],
                    text=text,
                    output_path=str(output),
                    emo_vector=[0, 0, 0, 0, 0, 0, 0, 0.8],
                    use_random=False,
                    verbose=True,
                )
            except Exception as exc:
                status = "failed"
                error = str(exc)
        row = base_row("IndexTTS2 original", "indextts2_pack_best2", text_id, text, output, ref_mode="pack")
        row.update({"seconds": time.perf_counter() - started, "status": status, "error": error})
        if status != "ok" or not output.exists():
            row["output"] = ""
        rows.append(row)
        print(f"IndexTTS2 {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def run_gpt() -> list[dict]:
    import soundfile as sf
    import torch
    import torchaudio

    repo = ROOT / "external" / "GPT-SoVITS"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "GPT_SoVITS"))
    os.environ.setdefault("is_half", "False")
    os.environ.setdefault("version", "v2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    (repo / "GPT_SoVITS" / "pretrained_models" / "fast_langdetect").mkdir(parents=True, exist_ok=True)

    def soundfile_load(path: str):
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T), sr

    torchaudio.load = soundfile_load
    from tools.i18n.i18n import I18nAuto
    from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

    i18n = I18nAuto()
    change_gpt_weights(
        gpt_path=str(
            repo
            / "GPT_SoVITS"
            / "pretrained_models"
            / "gsv-v2final-pretrained"
            / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        )
    )
    change_sovits_weights(
        sovits_path=str(repo / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s2G2333k.pth")
    )
    rows = []
    for text_id, text in TESTS:
        output = AUDIO / "gpt_sovits_v2_single_ref" / f"{text_id}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status = "ok"
        error = ""
        if not output.exists():
            try:
                result = list(
                    get_tts_wav(
                        ref_wav_path=str(SINGLE_REF_AUDIO),
                        prompt_text=SINGLE_REF_TEXT,
                        prompt_language=i18n("中文"),
                        text=text,
                        text_language=i18n("中文"),
                        top_p=1,
                        temperature=1,
                    )
                )
                if not result:
                    raise RuntimeError("GPT-SoVITS returned no audio")
                sr, audio = result[-1]
                sf.write(output, audio, sr)
            except Exception as exc:
                status = "failed"
                error = str(exc)
        row = base_row("GPT-SoVITS v2 original single-ref", "gpt_sovits_v2_single_ref", text_id, text, output, ref_mode="single")
        row.update({"seconds": time.perf_counter() - started, "status": status, "error": error})
        if status != "ok" or not output.exists():
            row["output"] = ""
        rows.append(row)
        print(f"GPT-SoVITS {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def run_qwen() -> list[dict]:
    from mlx_audio.tts.generate import generate_audio
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit")
    instruct = "台湾年轻女生，声音温柔清亮，低卷舌，语气自然，像真人访谈口吻。"
    rows = []
    for text_id, text in TESTS:
        output_dir = AUDIO / "qwen3_1p7b_ref_attempt_single_ref"
        output = output_dir / f"{text_id}.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status = "ok"
        error = ""
        if not output.exists():
            try:
                generate_audio(
                    text=text,
                    model=model,
                    instruct=instruct,
                    ref_audio=str(SINGLE_REF_AUDIO),
                    ref_text=SINGLE_REF_TEXT,
                    lang_code="zh",
                    output_path=str(output_dir),
                    file_prefix=text_id,
                    audio_format="wav",
                    verbose=False,
                )
                generated = output_dir / f"{text_id}_000.wav"
                if not generated.exists():
                    matches = sorted(output_dir.glob(f"{text_id}*.wav"))
                    if not matches:
                        raise RuntimeError("Qwen generated no wav")
                    generated = matches[-1]
                shutil.copy2(generated, output)
            except Exception as exc:
                status = "failed"
                error = str(exc)
        row = base_row("Qwen3 1.7B VoiceDesign ref attempt", "qwen3_1p7b_ref_attempt_single_ref", text_id, text, output, ref_mode="single")
        row.update({"seconds": time.perf_counter() - started, "status": status, "error": error})
        if status != "ok" or not output.exists():
            row["output"] = ""
        rows.append(row)
        print(f"Qwen {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def make_zipvoice():
    import sherpa_onnx

    model_dir = ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min"
    vocoder = ROOT / "models" / "sherpa" / "vocos_24khz.onnx"
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=str(model_dir / "tokens.txt"),
                encoder=str(model_dir / "encoder.int8.onnx"),
                decoder=str(model_dir / "decoder.int8.onnx"),
                data_dir=str(model_dir / "espeak-ng-data"),
                lexicon=str(model_dir / "lexicon.txt"),
                vocoder=str(vocoder),
            ),
            debug=False,
            num_threads=4,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError("invalid ZipVoice config")
    return sherpa_onnx.OfflineTts(config)


def run_sherpa_zipvoice() -> list[dict]:
    import librosa
    import sherpa_onnx
    import soundfile as sf

    pack = pack_best2()
    tts = make_zipvoice()
    ref_audio, sample_rate = librosa.load(pack["pack_audio"], sr=None)
    rows = []
    for text_id, text in TESTS:
        output = AUDIO / "sherpa_zipvoice_zhmin_16step" / f"{text_id}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status = "ok"
        error = ""
        if not output.exists():
            try:
                gen_config = sherpa_onnx.GenerationConfig()
                gen_config.reference_audio = ref_audio
                gen_config.reference_sample_rate = sample_rate
                gen_config.reference_text = pack["ref_text"]
                gen_config.num_steps = 16
                gen_config.extra["min_char_in_sentence"] = "20"
                audio = tts.generate(text, gen_config)
                if len(audio.samples) == 0:
                    raise RuntimeError("empty audio")
                sf.write(str(output), audio.samples, audio.sample_rate, subtype="PCM_16")
            except Exception as exc:
                status = "failed"
                error = str(exc)
        row = base_row("Sherpa ZipVoice int8 zh-min original 16-step", "sherpa_zipvoice_zhmin_16step", text_id, text, output, ref_mode="pack")
        row.update({"seconds": time.perf_counter() - started, "status": status, "error": error, "num_steps": 16})
        if status != "ok" or not output.exists():
            row["output"] = ""
        rows.append(row)
        print(f"ZipVoice16 {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def run_pocket() -> list[dict]:
    import librosa
    import sherpa_onnx
    import soundfile as sf

    model_dir = ROOT / "models" / "sherpa" / "sherpa-onnx-pocket-tts-int8-2026-01-26"
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            pocket=sherpa_onnx.OfflineTtsPocketModelConfig(
                lm_flow=str(model_dir / "lm_flow.int8.onnx"),
                lm_main=str(model_dir / "lm_main.int8.onnx"),
                encoder=str(model_dir / "encoder.onnx"),
                decoder=str(model_dir / "decoder.int8.onnx"),
                text_conditioner=str(model_dir / "text_conditioner.onnx"),
                vocab_json=str(model_dir / "vocab.json"),
                token_scores_json=str(model_dir / "token_scores.json"),
            ),
            debug=False,
            num_threads=4,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError("invalid PocketTTS config")
    tts = sherpa_onnx.OfflineTts(config)
    pack = pack_best2()
    ref_audio, sample_rate = librosa.load(pack["pack_audio"], sr=tts.sample_rate)
    rows = []
    for text_id, text in TESTS:
        output = AUDIO / "sherpa_pocket_int8_5step" / f"{text_id}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status = "ok"
        error = ""
        if not output.exists():
            try:
                gen_config = sherpa_onnx.GenerationConfig()
                gen_config.reference_audio = ref_audio
                gen_config.reference_sample_rate = sample_rate
                gen_config.num_steps = 5
                audio = tts.generate(text, gen_config)
                if len(audio.samples) == 0:
                    raise RuntimeError("empty audio")
                sf.write(str(output), audio.samples, audio.sample_rate, subtype="PCM_16")
            except Exception as exc:
                status = "failed"
                error = str(exc)
        row = base_row("Sherpa PocketTTS int8 original", "sherpa_pocket_int8_5step", text_id, text, output, ref_mode="pack")
        row.update({"seconds": time.perf_counter() - started, "status": status, "error": error, "num_steps": 5})
        if status != "ok" or not output.exists():
            row["output"] = ""
        rows.append(row)
        print(f"PocketTTS {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def run_chatterbox() -> list[dict]:
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS

    pack = pack_best2()
    device = "mps"
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
    except Exception:
        device = "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)
    sample_rate = getattr(model, "sr", 24000)
    rows = []
    for text_id, text in TESTS:
        output = AUDIO / "chatterbox_tts_original" / f"{text_id}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status = "ok"
        error = ""
        if not output.exists():
            try:
                wav = model.generate(
                    text,
                    audio_prompt_path=pack["pack_audio"],
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    temperature=0.8,
                )
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
                ta.save(str(output), wav.cpu(), sample_rate)
            except Exception as exc:
                status = "failed"
                error = str(exc)
        row = base_row("ChatterboxTTS original", "chatterbox_tts_original", text_id, text, output, ref_mode="pack")
        row.update({"seconds": time.perf_counter() - started, "status": status, "error": error, "device": device})
        if status != "ok" or not output.exists():
            row["output"] = ""
        rows.append(row)
        print(f"Chatterbox {text_id}: {row['seconds']:.2f}s {status}")
    return rows


RUNNERS = {
    "cosy": run_cosy,
    "f5": run_f5,
    "index": run_index,
    "gpt": run_gpt,
    "qwen": run_qwen,
    "sherpa_zipvoice": run_sherpa_zipvoice,
    "pocket": run_pocket,
    "chatterbox": run_chatterbox,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(RUNNERS))
    args = parser.parse_args()
    rows = RUNNERS[args.model]()
    write_rows(args.model, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
