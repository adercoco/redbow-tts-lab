# ZipVoice App Noise Diagnosis

Date: 2026-06-07

## Symptom

The live iPhone app generation sounded much more electronic/noisy than the fixed
audio in the listening report.

## Confirmed Cause

The installed app was using `vocos_24khz_qdq_all_int8.onnx`.

That candidate was created only to avoid the iOS ONNX Runtime crash caused by
the report vocoder's `ConvInteger` nodes. It passed the runtime smoke test, but
it was not calibrated with a real ZipVoice mel corpus and produced strong
high-frequency artifacts on some sentences.

## Measured Evidence

Same sentence set, same ZipVoice encoder/decoder, same prompt audio/text,
4-step generation, mild clean postprocess.

For sentence 3:

- Report fixed `qvocoder_clean_mild`: `hf_over_8k_db = -46.17`
- Current app QDQ-all: `hf_over_8k_db = -11.15`
- MatMul qint8 candidate: `hf_over_8k_db = -37.89`
- fp32 Vocos candidate: `hf_over_8k_db = -32.30`

The QDQ-all high-frequency energy was therefore far too high, matching the
reported electronic noise.

Generated comparison audio:

`/Users/ader/Documents/App/tmp/app_vs_report_zipvoice_diagnosis_v1`

## Fix Applied

Rejected `vocos_24khz_qdq_all_int8.onnx` for the app.

Switched app model pack to:

`vocos_24khz_matmul_qint8.onnx`

Why this candidate:

- Runs on iOS ONNX Runtime.
- Keeps the app small.
- Avoids the severe high-frequency artifact seen in QDQ-all.
- Still does not exactly match the report's dynamic qint8 vocoder.

## Installed Build

Installed on iPhone 17 Pro Max:

- Bundle id: `app.redbow.voice`
- App size: about `176MB`
- ZipVoice model pack: about `143MB`
- Model: Cosy→ZipVoice 4-step
- Vocos: MatMul-only qint8
- Runtime self-test: succeeded on device
- Device generation time for self-test sentence: `27.3s`

## Remaining Gap

The report fixed audio used `vocos_24khz_dynamic_qint8.onnx`, but that graph
contains `ConvInteger`, which crashes with the current iOS ONNX Runtime build.

To get closer to the report sound, the next real options are:

1. Re-export the dynamic qint8 vocoder into an iOS-supported graph using real
   calibration data.
2. Use fp32 Vocos in the app as a higher-quality fallback, accepting a larger
   package.
3. Build/use an iOS ONNX Runtime that supports the needed quantized conv ops.
4. Add deterministic seed or multi-sample selection if ZipVoice generation drift
   remains the main cause of report/live mismatch.
