# Piper Step1000 FP32 Mobile Baseline

Goal: phone-fast Taiwan Mandarin gentle-female TTS student baseline.

## Files

| File | Purpose |
|---|---|
| `model.onnx` | Stable FP32 Piper/VITS ONNX runtime model. |
| `model.onnx.json` | Piper voice config and phoneme id map. |
| `reference_fast_frontend.py` | Python reference for the lightweight pypinyin -> phoneme-id frontend. |
| `reference_phoneme_mapping.py` | Reference mapping used to build the phoneme-id training dataset. |

## Runtime Path

1. Normalize text.
2. Convert text to pinyin tone3 with a lightweight app-native frontend.
3. Convert pinyin to Piper Chinese phonemes.
4. Convert phonemes to ids using `model.onnx.json`.
5. Run `model.onnx` with ONNX Runtime.

Inputs:

| Name | Type | Shape |
|---|---|---|
| `input` | int64 | `[1, phoneme_count]` |
| `input_lengths` | int64 | `[1]` |
| `scales` | float32 | `[3]`, normally `[0.667, 1.0, 0.8]` |

## Current Benchmark

Measured on this Mac with the fast phoneme-id frontend:

| Metric | Value |
|---|---:|
| Model size | 60.4MB |
| Load time | 0.298s |
| Sample A | 0.110s generation for 4.551s audio, RTF 0.024 |
| Sample B | 0.147s generation for 6.177s audio, RTF 0.024 |

## Quantization Notes

| Candidate | Size | Status |
|---|---:|---|
| FP32 | 60.4MB | Stable baseline. |
| Dynamic QInt8, all supported ops | 18.4MB | Not usable in current ONNX Runtime CPU: `ConvInteger` kernel missing. |
| MatMul/Gemm QInt8 only | 60.8MB | Runs, but does not reduce size. |
| FP16 | 31.0MB | Not usable in current ONNX Runtime CPU: runtime `Reshape` error. |
| Static QDQ INT8 | n/a | Current ORT quantizer hits a Softmax calibration range bug on this graph. |

Next deployment step: test FP32 baseline in iOS/Android ONNX Runtime first, then revisit int8/float16 with target-specific execution providers such as CoreML, NNAPI, or QNN.
