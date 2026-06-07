# Golden Teacher: cosyvoice2_clear_best2_line04_v1

User-selected golden teacher sample for the Taiwan Mandarin low-r TTS project.

- Model: `CosyVoice2-0.5B`
- Source pack: `clear_best2_7s`
- Source text id: `line_04`
- Text: `如果你愿意的话，我们等一下再一起确认一次。`
- Speaker score: `0.7413` CAMPPlus cosine against reference
- Golden audio: `golden_teacher.wav`
- Raw generated audio: `golden_teacher.raw.wav`
- Reference audio: `reference_clear_best2_7s.wav`

Use this as the fixed target voice anchor for future teacher corpus generation and ZipVoice distillation. Do not silently replace it with a newly generated Cosy sample.
