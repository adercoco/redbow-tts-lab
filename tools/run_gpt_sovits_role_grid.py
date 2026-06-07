#!/usr/bin/env python3
"""Run a GPT-SoVITS role grid while loading model weights once."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external" / "GPT-SoVITS"
GPT_MODEL = REPO / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
SOVITS_MODEL = REPO / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s2G2333k.pth"


def soundfile_load(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T), sr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "GPT_SoVITS"))
    os.environ.setdefault("is_half", "False")
    os.environ.setdefault("version", "v2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    (REPO / "GPT_SoVITS" / "pretrained_models" / "fast_langdetect").mkdir(parents=True, exist_ok=True)
    torchaudio.load = soundfile_load

    from tools.i18n.i18n import I18nAuto
    from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

    i18n = I18nAuto()
    change_gpt_weights(gpt_path=str(GPT_MODEL))
    change_sovits_weights(sovits_path=str(SOVITS_MODEL))

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = []
    for item in plan:
        output = Path(item["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status = "ok"
        error = ""
        try:
            result = list(
                get_tts_wav(
                    ref_wav_path=item["ref_audio"],
                    prompt_text=item["ref_text"],
                    prompt_language=i18n("中文"),
                    text=item["text"],
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
        rows.append(
            {
                **item,
                "seconds": time.perf_counter() - started,
                "status": status,
                "error": error,
                "output": str(output) if output.exists() else "",
            }
        )
        print(f"{item['role_id']} {item['text_id']}: {rows[-1]['seconds']:.2f}s {status}")

    Path(args.results).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
