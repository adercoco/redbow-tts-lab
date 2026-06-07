#!/usr/bin/env python3
"""Generate CosyVoice2 auditions for authorized Conan character references."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torchaudio


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "conan_authorized_voice_refs_v1"
PACKS = BASE / "reference_packs" / "manifest.json"
OUT = BASE / "cosy_clone_audition_v1"
AUDIO = OUT / "audio"
COSY_ROOT = ROOT / "external" / "CosyVoice"

TESTS = {
    "haibara": [
        ("calm", "你先冷静一点。事情还没有到最糟的情况。"),
        ("confirm", "如果你愿意的话，我们等一下再一起确认一次。"),
        ("clue", "我只是觉得，这个线索不能太早下结论。"),
        ("soft", "不用担心啦，我会在旁边帮你看着。"),
    ],
    "conan": [
        ("truth", "真相只有一个。我已经知道答案了。"),
        ("odd", "等一下，这里有个地方很奇怪。"),
        ("clear", "不用担心，我一定会把事情查清楚。"),
        ("hurry", "现在还不能放弃，我们再找一次线索。"),
    ],
    "agasa": [
        ("invention", "这个发明虽然看起来普通，其实很厉害喔。"),
        ("explain", "你们两个先别急，听我慢慢说明。"),
        ("useful", "哈哈，这次的机关应该会派上用场。"),
        ("careful", "不过使用的时候要小心一点，别太勉强。"),
    ],
}


def load_packs() -> list[dict]:
    return json.loads(PACKS.read_text(encoding="utf-8"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(COSY_ROOT))
    sys.path.insert(0, str(COSY_ROOT / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(model_dir=str(COSY_ROOT / "pretrained_models" / "CosyVoice2-0.5B"))
    rows: list[dict[str, object]] = []
    for pack in load_packs():
        role = str(pack["role"])
        for text_id, text in TESTS[role]:
            output = AUDIO / role / f"{text_id}.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            status = "ok"
            error = ""
            started = time.perf_counter()
            if not output.exists():
                try:
                    for index, item in enumerate(
                        model.inference_zero_shot(
                            text,
                            str(pack["ref_text"]),
                            str(pack["pack_audio"]),
                            stream=False,
                        )
                    ):
                        if index == 0:
                            torchaudio.save(str(output), item["tts_speech"], model.sample_rate)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "role": role,
                    "pack_id": pack["pack_id"],
                    "ref_audio": pack["pack_audio"],
                    "ref_text": pack["ref_text"],
                    "ref_files": pack["ref_files"],
                    "ref_duration": pack["duration"],
                    "text_id": text_id,
                    "text": text,
                    "output": str(output) if status == "ok" else "",
                    "seconds": elapsed,
                    "status": status,
                    "error": error,
                    "model": "CosyVoice2-0.5B zero-shot clone",
                }
            )
            print(f"{role} {text_id}: {elapsed:.2f}s {status}")

    result_path = OUT / "cosy_clone_results.json"
    result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
