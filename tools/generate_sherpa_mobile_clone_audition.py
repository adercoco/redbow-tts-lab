#!/usr/bin/env python3
"""Generate mobile-oriented Sherpa-ONNX clone auditions from reference packs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import librosa
import sherpa_onnx
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
PACKS = BASE / "reference_packs_v1" / "manifest.json"
OUT = BASE / "clone_audition_v3_dataset_models"
AUDIO = OUT / "audio"
MODELS = ROOT / "models" / "sherpa"

TESTS = [
    ("tw_soft", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("tw_calm", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
    ("tw_decide", "我觉得这件事情可以慢慢来，不需要马上决定。"),
    ("tw_confirm", "如果你愿意的话，我们等一下再一起确认一次。"),
]


def load_packs() -> list[dict]:
    packs = json.loads(PACKS.read_text(encoding="utf-8"))
    return [pack for pack in packs if pack.get("pack_audio") and pack.get("pack_id") in {"pack_best2_7s", "pack_best3_11s"}]


def make_zipvoice(model_dir: Path, vocoder: Path, *, num_threads: int = 4) -> sherpa_onnx.OfflineTts:
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
            num_threads=num_threads,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError(f"Invalid ZipVoice config: {model_dir}")
    return sherpa_onnx.OfflineTts(config)


def make_pocket(model_dir: Path, *, num_threads: int = 4) -> sherpa_onnx.OfflineTts:
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
            num_threads=num_threads,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError(f"Invalid PocketTTS config: {model_dir}")
    return sherpa_onnx.OfflineTts(config)


def base_row(family: str, pack: dict, text_id: str, text: str, output: Path) -> dict:
    return {
        "family": family,
        "candidate_id": f"{family.lower().replace(' ', '_').replace('-', '_').replace('/', '_')}_{pack['pack_id']}",
        "pack_id": pack["pack_id"],
        "ref_id": pack["pack_id"],
        "ref_audio": pack["pack_audio"],
        "ref_text": pack["ref_text"],
        "ref_files": pack["ref_files"],
        "text_id": text_id,
        "text": text,
        "output": str(output),
    }


def run_zipvoice(rows: list[dict], packs: list[dict]) -> None:
    candidates = [
        {
            "family": "Sherpa ZipVoice int8 zh-min 8-step",
            "dir": MODELS / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min",
            "vocoder": MODELS / "vocos_24khz.onnx",
            "steps": 8,
            "slug": "sherpa_zipvoice_zhmin_8step",
        },
        {
            "family": "Sherpa ZipVoice int8 zh-min 4-step",
            "dir": MODELS / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min",
            "vocoder": MODELS / "vocos_24khz.onnx",
            "steps": 4,
            "slug": "sherpa_zipvoice_zhmin_4step",
        },
        {
            "family": "Sherpa ZipVoice int8 zh-min 4-step qint8 vocoder",
            "dir": MODELS / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min",
            "vocoder": MODELS / "vocos_24khz_dynamic_qint8.onnx",
            "steps": 4,
            "slug": "sherpa_zipvoice_zhmin_4step_qvocos",
        },
    ]
    for candidate in candidates:
        tts = make_zipvoice(candidate["dir"], candidate["vocoder"])
        for pack in packs:
            ref_audio, sample_rate = librosa.load(pack["pack_audio"], sr=None)
            for text_id, text in TESTS:
                output = AUDIO / candidate["slug"] / pack["pack_id"] / f"{text_id}.wav"
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
                        gen_config.num_steps = candidate["steps"]
                        gen_config.extra["min_char_in_sentence"] = "20"
                        audio = tts.generate(text, gen_config)
                        if len(audio.samples) == 0:
                            raise RuntimeError("empty audio")
                        sf.write(str(output), audio.samples, audio.sample_rate, subtype="PCM_16")
                    except Exception as exc:
                        status = "failed"
                        error = str(exc)
                row = base_row(candidate["family"], pack, text_id, text, output)
                row.update(
                    {
                        "seconds": time.perf_counter() - started,
                        "status": status,
                        "error": error,
                        "num_steps": candidate["steps"],
                        "model_dir": str(candidate["dir"]),
                        "vocoder": str(candidate["vocoder"]),
                    }
                )
                if status != "ok" or not output.exists():
                    row["output"] = ""
                rows.append(row)
                print(f"{candidate['family']} {pack['pack_id']} {text_id}: {row['seconds']:.2f}s {status}")


def run_pocket(rows: list[dict], packs: list[dict]) -> None:
    model_dir = MODELS / "sherpa-onnx-pocket-tts-int8-2026-01-26"
    tts = make_pocket(model_dir)
    family = "Sherpa PocketTTS int8 5-step"
    for pack in packs:
        ref_audio, sample_rate = librosa.load(pack["pack_audio"], sr=tts.sample_rate)
        for text_id, text in TESTS:
            output = AUDIO / "sherpa_pocket_int8_5step" / pack["pack_id"] / f"{text_id}.wav"
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
            row = base_row(family, pack, text_id, text, output)
            row.update(
                {
                    "seconds": time.perf_counter() - started,
                    "status": status,
                    "error": error,
                    "num_steps": 5,
                    "model_dir": str(model_dir),
                    "license_note": "non-commercial model pack",
                }
            )
            if status != "ok" or not output.exists():
                row["output"] = ""
            rows.append(row)
            print(f"{family} {pack['pack_id']} {text_id}: {row['seconds']:.2f}s {status}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    packs = load_packs()
    run_zipvoice(rows, packs)
    try:
        run_pocket(rows, packs)
    except Exception as exc:
        rows.append({"family": "Sherpa PocketTTS int8 5-step", "status": "failed", "error": str(exc)})
        print(f"Sherpa PocketTTS failed: {exc}")
    result_path = OUT / "sherpa_mobile_results.json"
    result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
