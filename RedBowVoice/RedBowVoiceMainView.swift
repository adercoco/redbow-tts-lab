import SwiftUI

struct RedBowVoiceMainView: View {
    @StateObject private var engine = VoiceEngine()
    @StateObject private var speech = SpeechInputController()
    @State private var recognizedText = ""
    @State private var isPressing = false
    @State private var didRunRecordingSelfTest = false
    @State private var didRunTTSSelfTest = false
    @State private var selectedPreset = VoicePreset.appModels[0]

    private struct SampleButtonItem: Identifiable {
        let id: String
        let title: String
        let text: String
        let audioPath: String?
    }

    private let zipVoiceSamples = [
        SampleButtonItem(
            id: "zip-report-calm",
            title: "冷靜一點",
            text: "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
            audioPath: nil
        ),
        SampleButtonItem(
            id: "zip-report-taiwan",
            title: "台灣尾音",
            text: "这样听起来有比较像台湾人吗？我希望尾音自然一点，不要太卷舌。",
            audioPath: nil
        ),
        SampleButtonItem(
            id: "zip-report-real",
            title: "真實聲音",
            text: "我想要的不是主播腔，也不是娃娃音，是聪明、温柔、真实的声音。",
            audioPath: nil
        ),
        SampleButtonItem(
            id: "zip-daily-confirm",
            title: "再確認",
            text: "如果你愿意的话，我们等一下再一起确认一次。",
            audioPath: nil
        )
    ]

    private let zipReportSamples = [
        SampleButtonItem(
            id: "zip-report-audio-01",
            title: "報告 01",
            text: "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
            audioPath: "Samples/ZipVoiceQvocoderClean/cosy_01.wav"
        ),
        SampleButtonItem(
            id: "zip-report-audio-02",
            title: "報告 02",
            text: "这样听起来有比较像台湾人吗？我希望尾音自然一点，不要太卷舌。",
            audioPath: "Samples/ZipVoiceQvocoderClean/cosy_02.wav"
        ),
        SampleButtonItem(
            id: "zip-report-audio-03",
            title: "報告 03",
            text: "我想要的不是主播腔，也不是娃娃音，是聪明、温柔、真实的声音。",
            audioPath: "Samples/ZipVoiceQvocoderClean/cosy_03.wav"
        )
    ]

    private let cosySamples = [
        SampleButtonItem(
            id: "cosy-line-01",
            title: "線索",
            text: "欸，你先别急啦，我们慢慢看，应该可以找到线索。",
            audioPath: "Samples/CosyGoldenTeacher/cosy_line_01.wav"
        ),
        SampleButtonItem(
            id: "cosy-line-02",
            title: "慢慢說",
            text: "没关系，你慢慢说，我在这边听，真的不用紧张。",
            audioPath: "Samples/CosyGoldenTeacher/cosy_line_02.wav"
        ),
        SampleButtonItem(
            id: "cosy-line-03",
            title: "慢慢決定",
            text: "我觉得这件事情可以慢慢来，不需要马上决定。",
            audioPath: "Samples/CosyGoldenTeacher/cosy_line_03.wav"
        ),
        SampleButtonItem(
            id: "cosy-line-04",
            title: "一起確認",
            text: "如果你愿意的话，我们等一下再一起确认一次。",
            audioPath: "Samples/CosyGoldenTeacher/cosy_line_04.wav"
        )
    ]

    private let indexTTS2Samples = [
        SampleButtonItem(
            id: "index-line-01",
            title: "慢慢來",
            text: "你先不要急，我们慢慢来，把事情一件一件处理好。",
            audioPath: "Samples/IndexTTS2Favorite/index_01.wav"
        ),
        SampleButtonItem(
            id: "index-line-02",
            title: "別擔心",
            text: "我刚刚看了一下，应该不是你的问题，你不用太担心。",
            audioPath: "Samples/IndexTTS2Favorite/index_02.wav"
        ),
        SampleButtonItem(
            id: "index-line-03",
            title: "我在聽",
            text: "没关系啦，你先讲，我在这边听，真的不用紧张。",
            audioPath: "Samples/IndexTTS2Favorite/index_03.wav"
        )
    ]

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.98, green: 0.08, blue: 0.13),
                    Color(red: 0.42, green: 0.02, blue: 0.04)
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            ScrollView {
                VStack(spacing: 18) {
                    VStack(spacing: 10) {
                        Text("紅色蝴蝶結")
                            .font(.system(size: 34, weight: .heavy))
                            .foregroundStyle(.white)
                        Text(statusText)
                            .font(.headline)
                            .foregroundStyle(.white.opacity(0.86))
                            .multilineTextAlignment(.center)
                            .frame(minHeight: 28)
                    }

                    bowButton

                    VStack(spacing: 8) {
                        Text(recognizedText.isEmpty ? "按住蝴蝶結開始說話" : recognizedText)
                            .font(.system(size: 22, weight: .semibold))
                            .foregroundStyle(.white)
                            .multilineTextAlignment(.center)
                            .lineLimit(4)
                            .minimumScaleFactor(0.72)
                            .frame(maxWidth: .infinity, minHeight: 78)
                            .padding(.horizontal, 22)

                        if !engine.lastLatencyText.isEmpty {
                            Text(engine.lastLatencyText)
                                .font(.footnote.weight(.medium))
                                .foregroundStyle(.white.opacity(0.72))
                        }
                    }

                    samplePanel(
                        title: "ZipVoice 即時生成",
                        subtitle: "按了會用手機內的 Cosy→ZipVoice 4-step 模型現場生成",
                        items: zipVoiceSamples,
                        action: synthesizeSample
                    )

                    samplePanel(
                        title: "ZipVoice 報告固定音檔",
                        subtitle: "報告裡的 qint8 Vocos + mild clean，用來跟上面現場生成對照",
                        items: zipReportSamples,
                        action: playZipReportSample
                    )

                    samplePanel(
                        title: "Cosy 老師試聽",
                        subtitle: "固定老師音檔，用來對比 Matcha 要追的聲音",
                        items: cosySamples,
                        action: playCosySample
                    )

                    samplePanel(
                        title: "IndexTTS2 最愛聲音",
                        subtitle: "你之前喜歡的大模型老師範例，只播放固定音檔",
                        items: indexTTS2Samples,
                        action: playIndexTTS2Sample
                    )
                }
                .padding(.horizontal, 22)
                .padding(.vertical, 24)
            }
        }
        .task {
            await loadSelectedModel()
            await runTTSSelfTestIfNeeded()
            await runRecordingSelfTestIfNeeded()
        }
    }

    private var bowButton: some View {
            RedBowButton(isPressed: isPressing || speech.isListening)
            .frame(width: 340, height: 230)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        guard !isPressing else { return }
                        isPressing = true
                        recognizedText = ""
                        engine.stopSpeaking()
                        Task {
                            await speech.start { text in
                                handleRecognizedText(text)
                            }
                        }
                    }
                    .onEnded { _ in
                        isPressing = false
                        speech.stopAndSubmit()
                    }
            )
            .accessibilityLabel("按住說話")
            .accessibilityIdentifier("redBowRecordButton")
    }

    private func samplePanel(
        title: String,
        subtitle: String,
        items: [SampleButtonItem],
        action: @escaping (SampleButtonItem) -> Void
    ) -> some View {
        VStack(spacing: 10) {
            VStack(spacing: 2) {
                Text(title)
                    .font(.footnote.weight(.bold))
                    .foregroundStyle(.white.opacity(0.82))
                    .frame(maxWidth: .infinity, alignment: .leading)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.62))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                ForEach(items) { item in
                    Button {
                        action(item)
                    } label: {
                        VStack(alignment: .leading, spacing: 5) {
                            Text(item.title)
                                .font(.system(size: 13, weight: .heavy))
                                .foregroundStyle(.white.opacity(0.92))
                            Text(item.text)
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(.white)
                                .multilineTextAlignment(.leading)
                                .lineLimit(3)
                                .minimumScaleFactor(0.8)
                        }
                        .frame(maxWidth: .infinity, minHeight: 76, alignment: .leading)
                        .padding(.horizontal, 12)
                        .background(.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .stroke(.white.opacity(0.22), lineWidth: 1)
                        )
                    }
                    .buttonStyle(.plain)
                    .disabled(speech.isListening)
                    .accessibilityLabel(item.title)
                }
            }
        }
    }

    private var statusText: String {
        switch speech.state {
        case .idle:
            switch engine.state {
            case .playing:
                return "正在用台灣中文唸回來"
            case let .failed(message):
                if selectedPreset.isPreviewOnly {
                    return "\(selectedPreset.name) 目前先聽樣本"
                }
                return message
            default:
                if selectedPreset.isPreviewOnly {
                    return "ZipVoice 尚未接手機生成，先用試聽樣本比較"
                }
                return "按住講話，放開後唸回同一句"
            }
        case .authorizing:
            return "正在請求麥克風權限"
        case .listening:
            return speech.partialText.isEmpty ? "正在聽你說話" : speech.partialText
        case let .failed(message):
            return message
        }
    }

    @MainActor
    private func loadSelectedModel() async {
        await engine.loadModel(preset: selectedPreset)
    }

    @MainActor
    private func runTTSSelfTestIfNeeded() async {
        guard CommandLine.arguments.contains("--self-test-tts"), !didRunTTSSelfTest else { return }
        didRunTTSSelfTest = true
        guard engine.isModelReady else {
            writeSelfTestLog("RED_BOW_TTS_SELF_TEST_FAILED model_not_ready")
            return
        }

        let text = selectedPreset.defaultPrompt
        writeSelfTestLog("RED_BOW_TTS_SELF_TEST_START text=\(text)")
        recognizedText = text
        await engine.synthesize(text: text, preset: selectedPreset)
        writeSelfTestLog("RED_BOW_TTS_SELF_TEST_DONE \(engine.lastLatencyText)")
    }

    private func writeSelfTestLog(_ line: String) {
        print(line)
        guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return
        }
        let logURL = documentsURL.appendingPathComponent("redbow-tts-self-test.log")
        let stampedLine = "\(Date()) \(line)\n"
        if let data = stampedLine.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: logURL.path),
               let handle = try? FileHandle(forWritingTo: logURL) {
                try? handle.seekToEnd()
                try? handle.write(contentsOf: data)
                try? handle.close()
            } else {
                try? data.write(to: logURL)
            }
        }
    }

    @MainActor
    private func runRecordingSelfTestIfNeeded() async {
        guard CommandLine.arguments.contains("--self-test-recording"), !didRunRecordingSelfTest else { return }
        didRunRecordingSelfTest = true

        try? await Task.sleep(nanoseconds: 700_000_000)

        for index in 1...3 {
            recognizedText = "模擬器錄音測試 \(index)/3"
            engine.stopSpeaking()

            await speech.start { text in
                handleRecognizedText(text)
            }

            try? await Task.sleep(nanoseconds: 1_000_000_000)
            speech.stopAndSubmit()
            try? await Task.sleep(nanoseconds: 500_000_000)
        }

        recognizedText = "模擬器錄音測試完成"
    }

    @MainActor
    private func handleRecognizedText(_ text: String) {
        recognizedText = text

        Task { @MainActor in
            if selectedPreset.isPreviewOnly {
                engine.markPresetUnavailable(selectedPreset)
            } else if engine.isModelReady {
                await engine.synthesize(text: text, preset: selectedPreset)
            } else {
                engine.speakBack(text)
            }
        }
    }

    @MainActor
    private func synthesizeSample(_ item: SampleButtonItem) {
        recognizedText = item.text
        engine.stopSpeaking()

        Task { @MainActor in
            if engine.isModelReady {
                await engine.synthesize(text: item.text, preset: selectedPreset)
            } else {
                await engine.loadModel(preset: selectedPreset)
                await engine.synthesize(text: item.text, preset: selectedPreset)
            }
        }
    }

    @MainActor
    private func playCosySample(_ item: SampleButtonItem) {
        guard let audioPath = item.audioPath else { return }
        recognizedText = item.text
        engine.playBundledSampleAudio(path: audioPath, title: item.title, group: "Cosy 老師")
    }

    @MainActor
    private func playZipReportSample(_ item: SampleButtonItem) {
        guard let audioPath = item.audioPath else { return }
        recognizedText = item.text
        engine.playBundledSampleAudio(path: audioPath, title: item.title, group: "ZipVoice 報告版")
    }

    @MainActor
    private func playIndexTTS2Sample(_ item: SampleButtonItem) {
        guard let audioPath = item.audioPath else { return }
        recognizedText = item.text
        engine.playBundledSampleAudio(path: audioPath, title: item.title, group: "IndexTTS2")
    }
}

private struct RedBowButton: View {
    let isPressed: Bool

    var body: some View {
        ZStack {
            Capsule()
                .fill(.black.opacity(0.16))
                .frame(width: isPressed ? 330 : 314, height: isPressed ? 164 : 154)
                .blur(radius: 12)
                .offset(y: 18)

            RedBowMark()
                .frame(width: isPressed ? 322 : 304, height: isPressed ? 186 : 176)
                .shadow(color: .black.opacity(0.35), radius: isPressed ? 24 : 18, y: 12)
                .scaleEffect(isPressed ? 0.96 : 1)
        }
        .animation(.spring(response: 0.22, dampingFraction: 0.72), value: isPressed)
    }
}

private struct RedBowMark: View {
    var body: some View {
        ZStack {
            BowLoop(side: .left)
                .fill(
                    LinearGradient(
                        colors: [Color(red: 1, green: 0.24, blue: 0.26), Color(red: 0.72, green: 0.02, blue: 0.05)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(BowLoop(side: .left).stroke(.black.opacity(0.3), lineWidth: 5))
                .overlay(BowFold(side: .left).stroke(.white.opacity(0.34), lineWidth: 4))
                .shadow(color: .black.opacity(0.22), radius: 8, x: 0, y: 6)

            BowLoop(side: .right)
                .fill(
                    LinearGradient(
                        colors: [Color(red: 1, green: 0.26, blue: 0.28), Color(red: 0.68, green: 0.01, blue: 0.04)],
                        startPoint: .topTrailing,
                        endPoint: .bottomLeading
                    )
                )
                .overlay(BowLoop(side: .right).stroke(.black.opacity(0.3), lineWidth: 5))
                .overlay(BowFold(side: .right).stroke(.white.opacity(0.34), lineWidth: 4))
                .shadow(color: .black.opacity(0.22), radius: 8, x: 0, y: 6)

            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [Color(red: 0.17, green: 0.02, blue: 0.03), Color.black],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(width: 74, height: 92)
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(.white.opacity(0.22), lineWidth: 3)
                )
                .overlay(
                    VStack(spacing: 6) {
                        Circle()
                            .fill(.white.opacity(0.84))
                            .frame(width: 10, height: 10)
                        ForEach(0..<3, id: \.self) { _ in
                            Capsule()
                                .fill(.white.opacity(0.66))
                                .frame(width: 34, height: 4)
                        }
                    }
                )
        }
    }
}

private enum BowSide {
    case left
    case right
}

private struct BowLoop: Shape {
    let side: BowSide

    func path(in rect: CGRect) -> Path {
        let midX = rect.midX
        let midY = rect.midY
        var path = Path()

        if side == .left {
            path.move(to: CGPoint(x: midX - 18, y: midY))
            path.addCurve(to: CGPoint(x: rect.minX + 38, y: rect.minY + 24),
                          control1: CGPoint(x: midX - 78, y: rect.minY + 10),
                          control2: CGPoint(x: rect.minX + 78, y: rect.minY - 12))
            path.addCurve(to: CGPoint(x: rect.minX + 8, y: midY),
                          control1: CGPoint(x: rect.minX - 8, y: rect.minY + 42),
                          control2: CGPoint(x: rect.minX - 12, y: rect.maxY - 42))
            path.addCurve(to: CGPoint(x: rect.minX + 38, y: rect.maxY - 24),
                          control1: CGPoint(x: rect.minX - 12, y: rect.maxY - 42),
                          control2: CGPoint(x: rect.minX - 8, y: rect.maxY - 24))
            path.addCurve(to: CGPoint(x: midX - 18, y: midY),
                          control1: CGPoint(x: rect.minX + 78, y: rect.maxY + 12),
                          control2: CGPoint(x: midX - 78, y: rect.maxY - 10))
        } else {
            path.move(to: CGPoint(x: midX + 18, y: midY))
            path.addCurve(to: CGPoint(x: rect.maxX - 38, y: rect.minY + 24),
                          control1: CGPoint(x: midX + 78, y: rect.minY + 10),
                          control2: CGPoint(x: rect.maxX - 78, y: rect.minY - 12))
            path.addCurve(to: CGPoint(x: rect.maxX - 8, y: midY),
                          control1: CGPoint(x: rect.maxX + 8, y: rect.minY + 42),
                          control2: CGPoint(x: rect.maxX + 12, y: rect.maxY - 42))
            path.addCurve(to: CGPoint(x: rect.maxX - 38, y: rect.maxY - 24),
                          control1: CGPoint(x: rect.maxX + 12, y: rect.maxY - 42),
                          control2: CGPoint(x: rect.maxX + 8, y: rect.maxY - 24))
            path.addCurve(to: CGPoint(x: midX + 18, y: midY),
                          control1: CGPoint(x: rect.maxX - 78, y: rect.maxY + 12),
                          control2: CGPoint(x: midX + 78, y: rect.maxY - 10))
        }

        path.closeSubpath()
        return path
    }
}

private struct BowFold: Shape {
    let side: BowSide

    func path(in rect: CGRect) -> Path {
        let midX = rect.midX
        let midY = rect.midY
        var path = Path()

        if side == .left {
            path.move(to: CGPoint(x: midX - 24, y: midY - 4))
            path.addCurve(to: CGPoint(x: rect.minX + 52, y: rect.minY + 36),
                          control1: CGPoint(x: midX - 78, y: rect.minY + 32),
                          control2: CGPoint(x: rect.minX + 76, y: rect.minY + 30))
            path.move(to: CGPoint(x: midX - 24, y: midY + 4))
            path.addCurve(to: CGPoint(x: rect.minX + 52, y: rect.maxY - 36),
                          control1: CGPoint(x: midX - 78, y: rect.maxY - 32),
                          control2: CGPoint(x: rect.minX + 76, y: rect.maxY - 30))
        } else {
            path.move(to: CGPoint(x: midX + 24, y: midY - 4))
            path.addCurve(to: CGPoint(x: rect.maxX - 52, y: rect.minY + 36),
                          control1: CGPoint(x: midX + 78, y: rect.minY + 32),
                          control2: CGPoint(x: rect.maxX - 76, y: rect.minY + 30))
            path.move(to: CGPoint(x: midX + 24, y: midY + 4))
            path.addCurve(to: CGPoint(x: rect.maxX - 52, y: rect.maxY - 36),
                          control1: CGPoint(x: midX + 78, y: rect.maxY - 32),
                          control2: CGPoint(x: rect.maxX - 76, y: rect.maxY - 30))
        }

        return path
    }
}

#Preview {
    RedBowVoiceMainView()
}
