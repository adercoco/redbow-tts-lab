import Foundation

private final class MatchaCStringBag {
    private var pointers: [UnsafeMutablePointer<CChar>] = []

    func add(_ string: String) -> UnsafePointer<CChar>! {
        guard let pointer = strdup(string) else { return nil }
        pointers.append(pointer)
        return UnsafePointer(pointer)
    }

    deinit {
        pointers.forEach { free($0) }
    }
}

struct SherpaMatchaModelPaths {
    let acousticModel: String
    let vocoder: String
    let tokens: String
    let lexicon: String
    let speed: Float
}

enum SherpaMatchaError: LocalizedError {
    case createTtsFailed
    case missingFrontend
    case generateFailed
    case writeFailed(String)

    var errorDescription: String? {
        switch self {
        case .createTtsFailed:
            return "建立 sherpa-onnx Matcha TTS 失敗。"
        case .missingFrontend:
            return "Matcha 中文 pinyin 前處理沒有載入。"
        case .generateFailed:
            return "Matcha 沒有生成有效音訊。"
        case let .writeFailed(path):
            return "寫出 Matcha 生成音檔失敗：\(path)"
        }
    }
}

final class SherpaMatchaRuntime: @unchecked Sendable {
    private let tts: OpaquePointer
    private let paths: SherpaMatchaModelPaths
    private let frontend: PinyinFrontend

    init(paths: SherpaMatchaModelPaths, frontend: PinyinFrontend? = PinyinFrontend.shared) throws {
        guard let frontend else {
            throw SherpaMatchaError.missingFrontend
        }
        self.paths = paths
        self.frontend = frontend

        let cStrings = MatchaCStringBag()
        let matcha = SherpaOnnxOfflineTtsMatchaModelConfig(
            acoustic_model: cStrings.add(paths.acousticModel),
            vocoder: cStrings.add(paths.vocoder),
            lexicon: cStrings.add(paths.lexicon),
            tokens: cStrings.add(paths.tokens),
            data_dir: cStrings.add(""),
            noise_scale: 0.5,
            length_scale: 1.0 / paths.speed,
            dict_dir: cStrings.add("")
        )

        let model = SherpaOnnxOfflineTtsModelConfig(
            vits: SherpaOnnxOfflineTtsVitsModelConfig(),
            num_threads: 2,
            debug: 0,
            provider: cStrings.add("cpu"),
            matcha: matcha,
            kokoro: SherpaOnnxOfflineTtsKokoroModelConfig(),
            kitten: SherpaOnnxOfflineTtsKittenModelConfig(),
            zipvoice: SherpaOnnxOfflineTtsZipvoiceModelConfig(),
            pocket: SherpaOnnxOfflineTtsPocketModelConfig(),
            supertonic: SherpaOnnxOfflineTtsSupertonicModelConfig()
        )

        var config = SherpaOnnxOfflineTtsConfig(
            model: model,
            rule_fsts: cStrings.add(""),
            max_num_sentences: 1,
            rule_fars: cStrings.add(""),
            silence_scale: 0.2
        )

        guard let created = SherpaOnnxCreateOfflineTts(&config) else {
            throw SherpaMatchaError.createTtsFailed
        }
        self.tts = created
    }

    deinit {
        SherpaOnnxDestroyOfflineTts(tts)
    }

    func synthesize(text: String, outputURL: URL) throws {
        let encodedText = try frontend.encodeForMatcha(text)
        let audio = encodedText.withCString { textPointer in
            var generation = SherpaOnnxGenerationConfig(
                silence_scale: 0.2,
                speed: paths.speed,
                sid: 0,
                reference_audio: nil,
                reference_audio_len: 0,
                reference_sample_rate: 0,
                reference_text: nil,
                num_steps: 0,
                extra: nil
            )

            return withUnsafePointer(to: &generation) { generationPointer in
                SherpaOnnxOfflineTtsGenerateWithConfig(
                    tts,
                    textPointer,
                    generationPointer,
                    nil,
                    nil
                )
            }
        }

        guard let audio else {
            throw SherpaMatchaError.generateFailed
        }
        defer { SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio) }

        let sampleCount = Int(audio.pointee.n)
        guard sampleCount > 0, let samples = audio.pointee.samples else {
            throw SherpaMatchaError.generateFailed
        }

        let processedSamples = Self.postProcess(
            Array(UnsafeBufferPointer(start: samples, count: sampleCount)),
            sampleRate: Float(audio.pointee.sample_rate)
        )

        let writeResult = outputURL.path.withCString { outputPathPointer in
            processedSamples.withUnsafeBufferPointer { buffer in
                SherpaOnnxWriteWave(
                    buffer.baseAddress,
                    Int32(buffer.count),
                    audio.pointee.sample_rate,
                    outputPathPointer
                )
            }
        }

        guard writeResult == 1 else {
            throw SherpaMatchaError.writeFailed(outputURL.path)
        }
    }

    private static func postProcess(_ input: [Float], sampleRate: Float) -> [Float] {
        guard !input.isEmpty else { return input }

        let mean = input.reduce(Float(0), +) / Float(input.count)
        let dcRemoved = input.map { $0 - mean }
        let highPassed = onePoleHighPass(dcRemoved, cutoff: 55, sampleRate: sampleRate)
        var filtered = onePoleLowPass(highPassed, cutoff: 10_300, sampleRate: sampleRate)

        let rms = sqrt(filtered.reduce(Float(0)) { $0 + $1 * $1 } / Float(filtered.count))
        let targetRms = pow(Float(10), Float(-17) / 20)
        if rms > 0.000_001 {
            let gain = min(targetRms / rms, 5.0)
            filtered = filtered.map { $0 * gain }
        }

        let limiterDrive: Float = 1.18
        filtered = filtered.map { tanh($0 * limiterDrive) / limiterDrive }

        let peakLimit = pow(Float(10), Float(-0.8) / 20)
        let peak = filtered.map { abs($0) }.max() ?? 0
        if peak > peakLimit {
            let gain = peakLimit / peak
            filtered = filtered.map { $0 * gain }
        }

        return filtered.map { max(-1, min(1, $0)) }
    }

    private static func onePoleHighPass(_ input: [Float], cutoff: Float, sampleRate: Float) -> [Float] {
        guard sampleRate > 0 else { return input }
        let dt = Float(1) / sampleRate
        let rc = Float(1) / (2 * .pi * cutoff)
        let alpha = rc / (rc + dt)
        var previousX: Float = 0
        var previousY: Float = 0

        return input.map { x in
            let y = alpha * (previousY + x - previousX)
            previousX = x
            previousY = y
            return y
        }
    }

    private static func onePoleLowPass(_ input: [Float], cutoff: Float, sampleRate: Float) -> [Float] {
        guard sampleRate > 0 else { return input }
        let dt = Float(1) / sampleRate
        let rc = Float(1) / (2 * .pi * cutoff)
        let alpha = dt / (rc + dt)
        var previousY: Float = 0

        return input.map { x in
            let y = previousY + alpha * (x - previousY)
            previousY = y
            return y
        }
    }
}
