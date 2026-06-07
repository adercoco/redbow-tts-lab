# ZipVoice Mainline

Primary voice:

`taiwan_mandarin_low_r`

Teacher profile:

`voice_profiles/taiwan_mandarin_low_r.json`

Teacher reference used by ZipVoice:

`distillation/taiwan_mandarin_low_r/teacher_qwen3_1p7b_seed/audio/seed_0001.wav`

Reference text:

```text
如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。
```

Model pack:

`models/sherpa/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia`

Vocoder:

`models/sherpa/vocos_24khz.onnx`

## Local Simulator Server

Run:

```bash
.venv-sherpa/bin/python tools/tts_server.py --provider zipvoice
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

Generate a WAV:

```bash
curl -X POST http://127.0.0.1:8765/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"如果你想试试这个声音，我现在可以讲给你听。","presetId":"taiwan-mandarin-low-r"}' \
  -o out.wav
```

The iOS simulator calls the same `/tts` endpoint through `RedBowVoice/VoiceEngine.swift`.

## Current Baseline

Report:

`distillation/taiwan_mandarin_low_r/reports/mobile_student_baseline/index.html`

Result:

ZipVoice int8 is the app-facing mainline because it nearly matched Qwen3 0.6B in proxy similarity while being much more practical for iOS/Android.

## Mobile Embedding Plan

1. Keep the local HTTP server for simulator and fast iteration.
2. Add sherpa-onnx iOS runtime and bundle these assets:
   - `tokens.txt`
   - `encoder.int8.onnx`
   - `decoder.int8.onnx`
   - `lexicon.txt`
   - `espeak-ng-data`
   - `vocos_24khz.onnx`
   - `seed_0001.wav`
   - reference text above
3. Mirror the Python `GenerationConfig` in Swift:
   - `num_steps = 4`
   - `min_char_in_sentence = 20`
   - reference audio sample rate from `seed_0001.wav`
4. Use the same model pack on Android through sherpa-onnx Kotlin/Java bindings.

## Next Work

- Restore or wire a dedicated RedBowVoice UI target. The current `RedBowVoice/ContentView.swift` contains a separate "睡 thread" demo, so UI work should be done carefully instead of overwriting it blindly.
- Add native sherpa-onnx iOS integration after simulator server flow is stable.
- Expand teacher corpus only if we decide to train a smaller single-voice model later.
