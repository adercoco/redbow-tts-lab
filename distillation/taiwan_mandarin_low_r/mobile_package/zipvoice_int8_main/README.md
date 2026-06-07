# ZipVoice INT8 Main Runtime

Decision: use Sherpa-ONNX ZipVoice int8 as the main phone TTS runtime path.

## Why This Is Main

ZipVoice is currently the best balance between voice similarity and mobile practicality.

| Metric | Value |
|---|---:|
| Runtime pack | Sherpa-ONNX ZipVoice int8 zh-min + vocoder |
| Pack size | about 180.7MB |
| Average generation | 1.15s / sentence on Mac |
| Average teacher-ref similarity | 0.924 |
| Peak RSS on Mac | 925.8MB |
| Best reference seed | `seed_0001` |

Piper step1000 remains useful as a self-distilled small-model backup, but its current voice quality is behind ZipVoice.

## Runtime Assets

Use these existing model files; do not duplicate them unless packaging for an app bundle.

| Asset | Path |
|---|---|
| Model dir | `/Users/ader/Documents/App/models/sherpa/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min` |
| Decoder | `/Users/ader/Documents/App/models/sherpa/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min/decoder.int8.onnx` |
| Encoder | `/Users/ader/Documents/App/models/sherpa/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min/encoder.int8.onnx` |
| Vocoder | `/Users/ader/Documents/App/models/sherpa/vocos_24khz.onnx` |
| Tokens | `/Users/ader/Documents/App/models/sherpa/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min/tokens.txt` |
| Lexicon | `/Users/ader/Documents/App/models/sherpa/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min/lexicon.txt` |
| eSpeak data | `/Users/ader/Documents/App/models/sherpa/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min/espeak-ng-data` |

## Reference Voice

The current default voice prompt should be `seed_0001`.

| File | Meaning |
|---|---|
| `audio/teacher_reference_seed_0001.wav` | Qwen3 teacher reference voice. |
| `audio/zipvoice_teacher_ref_seed_0001.wav` | ZipVoice output using teacher reference. |
| `audio/zipvoice_original_seed_0001.wav` | ZipVoice original female voice comparison. |

## Development Plan

1. Build the iOS/Android runtime around this ZipVoice pack.
2. Measure real-device cold start, peak memory, and sentence generation latency.
3. Cache reusable reference voice data if the runtime API exposes it.
4. Build a fixed Taiwan gentle-female reference set and keep `seed_0001` as default until a better reference wins.
5. Keep Piper step1000 FP32 as the self-distilled small-model backup branch.

## Open Risks

| Risk | Notes |
|---|---|
| Memory | Mac peak RSS is 925.8MB. Real phones may need aggressive unloading/caching decisions. |
| Latency | Mac average is 1.15s/sentence; phone CPU may be slower without acceleration. |
| Voice control | This is zero-shot reference prompting, not trained single-speaker distillation. |
| Package size | 180.7MB is acceptable for a prototype but large for some app distribution paths. |
