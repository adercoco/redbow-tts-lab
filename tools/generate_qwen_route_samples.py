from __future__ import annotations

import json
import resource
import time
import wave
from pathlib import Path

from mlx_audio.tts.generate import generate_audio, load_model


ROOT = Path("/Users/ader/Documents/App")
OUT = ROOT / "distillation/taiwan_mandarin_low_r/reports/teacher_to_zipvoice_route_v1/generated"
REF_AUDIO = ROOT / "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/reference_packs_v1/pack_best2_7s.wav"
REF_TEXT = "所以我当时就说,我想要做一张疗愈人的专辑。 开始当然就是我们的提案会议,我就提出了因为多年"

TEXTS = [
    ("daily_01", "等一下我先把资料整理好，晚点再跟你确认一次。"),
    ("daily_02", "今天先不要想太多，回家路上买杯热的，慢慢来就好。"),
    ("daily_03", "你刚刚那句我有听到，我觉得可以再温柔一点说。"),
]

MODELS = [
    ("qwen3_0p6b_base", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"),
    ("qwen3_1p7b_base", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"),
]


def peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw > 10_000_000:
        return raw / 1024 / 1024
    return raw / 1024


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, model_id in MODELS:
        model_dir = OUT / key
        model_dir.mkdir(parents=True, exist_ok=True)
        load_start = time.perf_counter()
        model = load_model(model_id)
        load_s = time.perf_counter() - load_start
        load_peak = peak_rss_mb()
        for sample_id, text in TEXTS:
            started = time.perf_counter()
            generate_audio(
                text=text,
                model=model,
                lang_code="zh",
                ref_audio=str(REF_AUDIO),
                ref_text=REF_TEXT,
                output_path=str(model_dir),
                file_prefix=sample_id,
                audio_format="wav",
                verbose=False,
            )
            output = model_dir / f"{sample_id}_000.wav"
            wall_s = time.perf_counter() - started
            rows.append(
                {
                    "family": key,
                    "model_id": model_id,
                    "sample_id": sample_id,
                    "text": text,
                    "output": str(output),
                    "load_s": load_s,
                    "load_peak_rss_mb": load_peak,
                    "wall_s": wall_s,
                    "audio_s": wav_seconds(output),
                    "rtf": wall_s / wav_seconds(output),
                    "peak_rss_mb": peak_rss_mb(),
                    "status": "ok",
                }
            )
            print(key, sample_id, f"{wall_s:.2f}s", output)
        del model

    (OUT / "qwen_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT / "qwen_results.json")


if __name__ == "__main__":
    main()
