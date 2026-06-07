import AVFoundation
import Foundation

@MainActor
final class VoiceEngine: NSObject, ObservableObject, AVAudioPlayerDelegate, AVSpeechSynthesizerDelegate {
    enum State: Equatable {
        case idle
        case generating
        case ready(URL)
        case playing
        case failed(String)
    }

    enum ModelLoadState: Equatable {
        case unloaded
        case loading
        case ready(String)
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var modelLoadState: ModelLoadState = .unloaded
    @Published private(set) var lastLatencyText = ""

    private var player: AVAudioPlayer?
    private var currentAudioURL: URL?
    private let speechSynthesizer = AVSpeechSynthesizer()
    private var loadedPresetId: String?
    private var loadedModelPack: ModelPack?
    private var loadedRuntime: LoadedRuntime?

    override init() {
        super.init()
        speechSynthesizer.delegate = self
    }

    var canReplay: Bool {
        if case .ready = state { return true }
        return false
    }

    var isModelReady: Bool {
        if case .ready = modelLoadState { return true }
        return false
    }

    func loadModel(preset: VoicePreset) async {
        if isModelReady, loadedPresetId == preset.id { return }

        modelLoadState = .loading

        do {
            let packURL = try bundledModelPackURL(for: preset)
            let packData = try Data(contentsOf: packURL)
            let pack = try JSONDecoder().decode(ModelPack.self, from: packData)
            let modelDir = packURL.deletingLastPathComponent()
            let sizeText = formatBytes(directorySize(modelDir))

            guard pack.available else {
                loadedPresetId = nil
                loadedModelPack = nil
                loadedRuntime = nil
                let message = pack.notes ?? "\(pack.displayName) 還沒匯出成手機可用模型。"
                modelLoadState = .failed(message)
                lastLatencyText = "\(preset.name) 目前是試聽樣本，還不能任意文字離線生成。"
                return
            }

            let runtime = try await Task.detached(priority: .userInitiated) {
                try LoadedRuntime(pack: pack, modelDir: modelDir)
            }.value

            loadedPresetId = preset.id
            loadedModelPack = pack
            loadedRuntime = runtime
            let stepText = pack.numSteps.map { " · \($0)-step" } ?? ""
            let backendText = pack.backend.map { " · \($0)" } ?? ""
            let detail = "\(pack.displayName) · \(sizeText)\(stepText)\(backendText)"
            modelLoadState = .ready(detail)
            lastLatencyText = "模型已載入：\(preset.name)"
        } catch {
            modelLoadState = .failed("找不到可用模型包：\(error.localizedDescription)")
        }
    }

    func playBundledPreviewAudio(path: String, preset: VoicePreset) {
        stopSpeaking()

        let previewURL = Bundle.main.bundleURL
            .appendingPathComponent("RedBowAssets")
            .appendingPathComponent(path)

        guard FileManager.default.fileExists(atPath: previewURL.path) else {
            state = .failed("找不到 \(preset.name) 的試聽音檔。")
            return
        }

        do {
            try play(url: previewURL)
            lastLatencyText = "\(preset.name) 試聽樣本；任意文字離線生成尚未匯出。"
        } catch {
            state = .failed(error.readableMessage)
        }
    }

    func playBundledSampleAudio(path: String, title: String, group: String) {
        stopSpeaking()

        let sampleURL = Bundle.main.bundleURL
            .appendingPathComponent("RedBowAssets")
            .appendingPathComponent(path)

        guard FileManager.default.fileExists(atPath: sampleURL.path) else {
            state = .failed("找不到 \(group) 音檔：\(title)")
            return
        }

        do {
            try play(url: sampleURL)
            lastLatencyText = "播放 \(group)：\(title)"
        } catch {
            state = .failed(error.readableMessage)
        }
    }

    func markPresetUnavailable(_ preset: VoicePreset) {
        stopSpeaking()
        let message = "\(preset.name) 目前只有試聽樣本，還不能把你剛講的句子離線生成。"
        state = .failed(message)
        lastLatencyText = "先切回 ZipVoice 可測錄音後生成；Matcha 等 mobile export 接上。"
    }

    func synthesize(text: String, preset: VoicePreset) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            state = .failed("先輸入一句要試聲的中文。")
            return
        }
        guard isModelReady else {
            state = .failed("先按「載入模型」。之後同一次開 app 就不用重載。")
            return
        }
        guard let runtime = loadedRuntime else {
            state = .failed("模型 metadata 已載入，但 native sherpa-onnx runtime 尚未建立。")
            return
        }
        do {
            try stopPlayback()
            state = .generating
            lastLatencyText = ""

            let startedAt = Date()
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("red-bow-tts-\(UUID().uuidString).wav")

            try await Task.detached(priority: .userInitiated) {
                try runtime.synthesize(text: trimmed, outputURL: outputURL)
            }.value

            lastLatencyText = String(format: "生成 %.1f 秒", Date().timeIntervalSince(startedAt))
            state = .ready(outputURL)
            try play(url: outputURL)
        } catch {
            state = .failed(error.readableMessage)
        }
    }

    func replay() {
        guard case let .ready(url) = state else { return }
        try? play(url: url)
    }

    func speakBack(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            state = .failed("沒有可播放的文字。")
            return
        }

        stopSpeaking()

        let utterance = AVSpeechUtterance(string: trimmed)
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-TW")
        utterance.rate = 0.48
        utterance.pitchMultiplier = 1.08
        utterance.volume = 1.0

        lastLatencyText = "目前先用 iOS 內建中文 TTS 驗證互動；之後接離線 clone 模型。"
        state = .playing
        speechSynthesizer.speak(utterance)
    }

    func stopSpeaking() {
        speechSynthesizer.stopSpeaking(at: .immediate)
        player?.stop()
        player = nil
    }

    func stopPlayback() throws {
        player?.stop()
        player = nil
    }

    func reset() {
        try? stopPlayback()
        lastLatencyText = ""
        state = .idle
    }

    private func bundledModelPackURL(for preset: VoicePreset) throws -> URL {
        if let directory = preset.modelPackDirectory {
            let presetCandidates = [
                "RedBowAssets/Models/\(directory)",
                "Models/\(directory)"
            ]

            if let url = presetCandidates.compactMap({ subdirectory in
                Bundle.main.url(
                    forResource: "model-pack",
                    withExtension: "json",
                    subdirectory: subdirectory
                )
            }).first {
                return url
            }

            throw TTSError.server("\(preset.name) 的 model-pack.json 不在 app bundle。")
        }

        let fallbackDirectories = [
            "QwenZipVoiceDistillInt8",
            "CosyTinyTrueDistillInt8",
            "ZipVoiceDistillInt8"
        ]
        let fallbackCandidates = fallbackDirectories.flatMap { directory in
            [
                "RedBowAssets/Models/\(directory)",
                "Models/\(directory)"
            ]
        }

        if let url = fallbackCandidates.compactMap({ subdirectory in
            Bundle.main.url(
                forResource: "model-pack",
                withExtension: "json",
                subdirectory: subdirectory
            )
        }).first {
            return url
        }

        throw TTSError.server("model-pack.json 不在 app bundle。")
    }

    private func directorySize(_ url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        return enumerator.compactMap { item -> Int64? in
            guard let fileURL = item as? URL,
                  let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey]) else {
                return nil
            }
            return Int64(values.fileSize ?? 0)
        }.reduce(0, +)
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let mb = Double(bytes) / 1_048_576.0
        return String(format: "%.1fMB", mb)
    }

    private func play(url: URL) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try session.setActive(true)

        let player = try AVAudioPlayer(contentsOf: url)
        player.delegate = self
        player.prepareToPlay()
        self.player = player
        currentAudioURL = url
        state = .playing
        player.play()
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            if case .playing = self.state, let url = self.currentAudioURL {
                self.state = .ready(url)
            }
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            if case .playing = self.state {
                self.state = .idle
            }
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            if case .playing = self.state {
                self.state = .idle
            }
        }
    }
}

private struct ModelPack: Decodable {
    let displayName: String
    let acousticModel: String?
    let encoder: String?
    let decoder: String?
    let vocoder: String?
    let tokens: String?
    let lexicon: String?
    let dataDir: String?
    let promptAudio: String?
    let promptText: String?
    let numSteps: Int?
    let guidanceScale: Float?
    let speed: Float?
    let backend: String?
    let available: Bool
    let notes: String?

    private enum CodingKeys: String, CodingKey {
        case displayName
        case acousticModel
        case encoder
        case decoder
        case vocoder
        case tokens
        case lexicon
        case dataDir
        case promptAudio
        case promptText
        case numSteps
        case guidanceScale
        case speed
        case backend
        case available
        case notes
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        displayName = try container.decode(String.self, forKey: .displayName)
        acousticModel = try container.decodeIfPresent(String.self, forKey: .acousticModel)
        encoder = try container.decodeIfPresent(String.self, forKey: .encoder)
        self.decoder = try container.decodeIfPresent(String.self, forKey: .decoder)
        vocoder = try container.decodeIfPresent(String.self, forKey: .vocoder)
        tokens = try container.decodeIfPresent(String.self, forKey: .tokens)
        lexicon = try container.decodeIfPresent(String.self, forKey: .lexicon)
        dataDir = try container.decodeIfPresent(String.self, forKey: .dataDir)
        promptAudio = try container.decodeIfPresent(String.self, forKey: .promptAudio)
        promptText = try container.decodeIfPresent(String.self, forKey: .promptText)
        numSteps = try container.decodeIfPresent(Int.self, forKey: .numSteps)
        guidanceScale = try container.decodeIfPresent(Float.self, forKey: .guidanceScale)
        speed = try container.decodeIfPresent(Float.self, forKey: .speed)
        backend = try container.decodeIfPresent(String.self, forKey: .backend)
        available = try container.decodeIfPresent(Bool.self, forKey: .available) ?? true
        notes = try container.decodeIfPresent(String.self, forKey: .notes)
    }

    func sherpaPaths(relativeTo modelDir: URL) throws -> SherpaZipVoiceModelPaths {
        func required(_ value: String?, _ name: String) throws -> String {
            guard let value, !value.isEmpty else {
                throw TTSError.server("model-pack.json 缺少 \(name)。")
            }
            return value
        }

        func pathInModelDir(_ value: String?, _ name: String) throws -> String {
            modelDir.appendingPathComponent(try required(value, name)).path
        }

        func bundlePath(_ relativePath: String?, _ name: String) throws -> String {
            let value = try required(relativePath, name)
            let url = Bundle.main.bundleURL.appendingPathComponent("RedBowAssets").appendingPathComponent(value)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw TTSError.server("找不到 \(name)：\(value)")
            }
            return url.path
        }

        return SherpaZipVoiceModelPaths(
            encoder: try pathInModelDir(encoder, "encoder"),
            decoder: try pathInModelDir(decoder, "decoder"),
            vocoder: try pathInModelDir(vocoder, "vocoder"),
            tokens: try pathInModelDir(tokens, "tokens"),
            lexicon: try pathInModelDir(lexicon, "lexicon"),
            dataDir: try pathInModelDir(dataDir, "dataDir"),
            promptAudio: try bundlePath(promptAudio, "promptAudio"),
            promptText: try required(promptText, "promptText"),
            numSteps: numSteps ?? 4,
            guidanceScale: guidanceScale ?? 3.0,
            speed: speed ?? 1.0
        )
    }

    func sherpaMatchaPaths(relativeTo modelDir: URL) throws -> SherpaMatchaModelPaths {
        func required(_ value: String?, _ name: String) throws -> String {
            guard let value, !value.isEmpty else {
                throw TTSError.server("model-pack.json 缺少 \(name)。")
            }
            return value
        }

        func pathInModelDir(_ value: String?, _ name: String) throws -> String {
            modelDir.appendingPathComponent(try required(value, name)).path
        }

        return SherpaMatchaModelPaths(
            acousticModel: try pathInModelDir(acousticModel, "acousticModel"),
            vocoder: try pathInModelDir(vocoder, "vocoder"),
            tokens: try pathInModelDir(tokens, "tokens"),
            lexicon: try pathInModelDir(lexicon, "lexicon"),
            speed: speed ?? 1.18
        )
    }
}

private enum LoadedRuntime: @unchecked Sendable {
    case zipvoice(SherpaZipVoiceRuntime)
    case matcha(SherpaMatchaRuntime)

    init(pack: ModelPack, modelDir: URL) throws {
        let backend = pack.backend?.lowercased() ?? ""
        if backend.contains("matcha") {
            self = .matcha(try SherpaMatchaRuntime(paths: pack.sherpaMatchaPaths(relativeTo: modelDir)))
        } else {
            self = .zipvoice(try SherpaZipVoiceRuntime(paths: pack.sherpaPaths(relativeTo: modelDir)))
        }
    }

    func synthesize(text: String, outputURL: URL) throws {
        switch self {
        case let .zipvoice(runtime):
            try runtime.synthesize(text: text, outputURL: outputURL)
        case let .matcha(runtime):
            try runtime.synthesize(text: text, outputURL: outputURL)
        }
    }
}

private struct TTSRequest: Encodable {
    let text: String
    let presetId: String
    let presetName: String
    let modelHint: String
    let modelPackDirectory: String?

    init(text: String, preset: VoicePreset) {
        self.text = text
        self.presetId = preset.id
        self.presetName = preset.name
        self.modelHint = preset.modelHint
        self.modelPackDirectory = preset.modelPackDirectory
    }
}

private enum TTSError: Error {
    case invalidResponse
    case server(String)
}

private extension Error {
    var readableMessage: String {
        if let error = self as? TTSError {
            switch error {
            case .invalidResponse:
                return "TTS 沒有回傳有效音訊。"
            case let .server(message):
                return message
            }
        }
        return localizedDescription
    }
}
