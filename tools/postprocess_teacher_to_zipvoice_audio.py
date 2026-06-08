from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


ROOT = Path("/Users/ader/Documents/App")
REPORT_DIR = ROOT / "distillation/taiwan_mandarin_low_r/reports/teacher_to_zipvoice_route_v1"
GENERATED = REPORT_DIR / "generated"
PROCESSED = REPORT_DIR / "processed"
REF_AUDIO = ROOT / "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/reference_packs_v1/pack_best2_7s.wav"
TARGET_RMS_DBFS = -19.0
PEAK_CEILING_DBFS = -1.0


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
    return 20.0 * math.log10(rms)


def peak_db(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x)) + 1e-12)
    return 20.0 * math.log10(peak)


def highpass(x: np.ndarray, sr: int, cutoff: float = 70.0) -> np.ndarray:
    sos = signal.butter(2, cutoff / (sr / 2.0), btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def spectral_clean(x: np.ndarray, sr: int) -> np.ndarray:
    nperseg = 1024 if sr >= 22050 else 512
    noverlap = nperseg // 2
    freqs, times, zxx = signal.stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary="zeros")
    mag = np.abs(zxx)
    phase = np.exp(1j * np.angle(zxx))
    energy = mag.mean(axis=0)
    cutoff = np.percentile(energy, 25)
    noise_frames = mag[:, energy <= cutoff]
    if noise_frames.shape[1] < 2:
        noise = np.percentile(mag, 20, axis=1, keepdims=True)
    else:
        noise = np.median(noise_frames, axis=1, keepdims=True)
    floor = 0.18 * mag
    cleaned = np.maximum(mag - 0.75 * noise, floor)
    _, y = signal.istft(cleaned * phase, fs=sr, nperseg=nperseg, noverlap=noverlap, input_onesided=True)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y[: len(x)].astype(np.float32)


def soft_gate(x: np.ndarray, frame: int = 512, hop: int = 128) -> np.ndarray:
    if len(x) < frame:
        return x
    padded = np.pad(x, (0, frame), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    frame_rms = np.sqrt(np.mean(windows**2, axis=1) + 1e-12)
    threshold = max(np.percentile(frame_rms, 18) * 1.6, 10 ** (-50 / 20))
    gains = np.clip((frame_rms / threshold) ** 1.2, 0.35, 1.0)
    gain_curve = np.interp(np.arange(len(x)), np.arange(len(gains)) * hop, gains)
    return (x * gain_curve).astype(np.float32)


def loudness_match_and_limit(x: np.ndarray) -> np.ndarray:
    current = rms_db(x)
    gain = 10 ** ((TARGET_RMS_DBFS - current) / 20.0)
    y = x * gain
    ceiling = 10 ** (PEAK_CEILING_DBFS / 20.0)
    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > ceiling:
        y = y * (ceiling / peak)
    # Very light soft clipping protects against inter-sample-ish overs while
    # keeping the timbre closer to the model output than hard clipping.
    y = np.tanh(y / ceiling) * ceiling
    return np.clip(y, -0.98, 0.98).astype(np.float32)


def process_one(src: Path, dst: Path) -> dict[str, float | str]:
    x, sr = read_audio(src)
    before = {"rms_db": rms_db(x), "peak_db": peak_db(x)}
    y = x - float(np.mean(x))
    y = highpass(y, sr)
    y = spectral_clean(y, sr)
    y = soft_gate(y)
    y = loudness_match_and_limit(y)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, sr, subtype="PCM_16")
    after = {"rms_db": rms_db(y), "peak_db": peak_db(y)}
    return {
        "source": str(src),
        "output": str(dst),
        "before_rms_db": round(before["rms_db"], 2),
        "before_peak_db": round(before["peak_db"], 2),
        "after_rms_db": round(after["rms_db"], 2),
        "after_peak_db": round(after["peak_db"], 2),
        "target_rms_db": TARGET_RMS_DBFS,
        "peak_ceiling_db": PEAK_CEILING_DBFS,
    }


def update_json(path: Path, report: list[dict]) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        src = Path(row["output"])
        rel = src.relative_to(GENERATED)
        dst = PROCESSED / rel
        stats = process_one(src, dst)
        row["raw_output"] = row["output"]
        row["output"] = str(dst)
        row["postprocess"] = stats
        report.append({"family": row["family"], "sample_id": row["sample_id"], **stats})
    processed_json = PROCESSED / path.name
    processed_json.parent.mkdir(parents=True, exist_ok=True)
    processed_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    report = []
    ref_stats = process_one(REF_AUDIO, PROCESSED / "reference" / "pack_best2_7s.wav")
    report.append({"family": "reference", "sample_id": "pack_best2_7s", **ref_stats})
    for name in ["qwen_results.json", "cosy_results.json", "zipvoice_results.json"]:
        update_json(GENERATED / name, report)
    (PROCESSED / "postprocess_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(PROCESSED / "postprocess_report.json")


if __name__ == "__main__":
    main()
