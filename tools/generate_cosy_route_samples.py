from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import torch
import torchaudio


ROOT = Path("/Users/ader/Documents/App")
OUT = ROOT / "distillation/taiwan_mandarin_low_r/reports/teacher_to_zipvoice_route_v1/generated"
COSY_ROOT = ROOT / "external/CosyVoice"
REF_AUDIO = ROOT / "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/reference_packs_v1/pack_best2_7s.wav"
REF_TEXT = "所以我当时就说,我想要做一张疗愈人的专辑。 开始当然就是我们的提案会议,我就提出了因为多年"

TEXTS = [
    ("daily_01", "等一下我先把资料整理好，晚点再跟你确认一次。"),
    ("daily_02", "今天先不要想太多，回家路上买杯热的，慢慢来就好。"),
    ("daily_03", "你刚刚那句我有听到，我觉得可以再温柔一点说。"),
]


def peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw > 10_000_000:
        return raw / 1024 / 1024
    return raw / 1024


def normalize_output(wav: torch.Tensor) -> torch.Tensor:
    wav = wav - wav.mean()
    rms = torch.sqrt(torch.mean(wav**2)).clamp_min(1e-8)
    target = 10 ** (-20.0 / 20.0)
    wav = wav * (target / rms)
    peak = wav.abs().max().clamp_min(1e-8)
    if peak > 0.96:
        wav = wav * (0.96 / peak)
    return wav.clamp(-0.98, 0.98)


def wav_seconds(path: Path) -> float:
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def main() -> None:
    sys.path.insert(0, str(COSY_ROOT))
    sys.path.insert(0, str(COSY_ROOT / "third_party/Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    out_dir = OUT / "cosyvoice2_0p5b"
    out_dir.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    model = AutoModel(model_dir=str(COSY_ROOT / "pretrained_models/CosyVoice2-0.5B"))
    load_s = time.perf_counter() - load_start
    load_peak = peak_rss_mb()
    rows = []
    for sample_id, text in TEXTS:
        output = out_dir / f"{sample_id}.wav"
        raw_output = out_dir / f"{sample_id}.raw.wav"
        started = time.perf_counter()
        first = None
        for index, item in enumerate(model.inference_zero_shot(text, REF_TEXT, str(REF_AUDIO), stream=False)):
            if index == 0:
                first = item["tts_speech"]
                torchaudio.save(str(raw_output), first, model.sample_rate)
                torchaudio.save(str(output), normalize_output(first), model.sample_rate)
                break
        if first is None:
            raise RuntimeError("Cosy generated no audio")
        wall_s = time.perf_counter() - started
        rows.append(
            {
                "family": "cosyvoice2_0p5b",
                "model_id": "CosyVoice2-0.5B",
                "sample_id": sample_id,
                "text": text,
                "output": str(output),
                "raw_output": str(raw_output),
                "load_s": load_s,
                "load_peak_rss_mb": load_peak,
                "wall_s": wall_s,
                "audio_s": wav_seconds(output),
                "rtf": wall_s / wav_seconds(output),
                "peak_rss_mb": peak_rss_mb(),
                "status": "ok",
            }
        )
        print("cosyvoice2_0p5b", sample_id, f"{wall_s:.2f}s", output)
    (OUT / "cosy_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT / "cosy_results.json")


if __name__ == "__main__":
    main()
