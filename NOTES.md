# 紅色蝴蝶結 Prototype Notes

Question: can we find a TTS/model path whose Chinese output is good enough before investing in production app polish?

Decision for this prototype:

- Build native iOS with SwiftUI and AVFoundation for the app shell.
- Use a localhost TTS bridge while testing in the simulator, because model quality is the main risk.
- Keep a stable `POST /tts` interface so F5-TTS, IndexTTS2, or a licensed model can be swapped in.
- Avoid cloning or naming real Chinese dub actors/public figures without consent. The presets are character-inspired voice styles, not biometric imitations.

Next production step:

- Replace the smoke-test macOS `say` provider in `tools/tts_server.py` with the best cloning/TTS provider after listening tests.
