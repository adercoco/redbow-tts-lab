import Foundation

private final class CStringBag {
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

struct SherpaZipVoiceModelPaths {
    let encoder: String
    let decoder: String
    let vocoder: String
    let tokens: String
    let lexicon: String
    let dataDir: String
    let promptAudio: String
    let promptText: String
    let numSteps: Int
    let guidanceScale: Float
    let speed: Float
}

enum SherpaZipVoiceError: LocalizedError {
    case createTtsFailed
    case readPromptFailed(String)
    case generateFailed
    case writeFailed(String)

    var errorDescription: String? {
        switch self {
        case .createTtsFailed:
            return "建立 sherpa-onnx ZipVoice TTS 失敗。"
        case let .readPromptFailed(path):
            return "讀不到 ZipVoice reference 音檔：\(path)"
        case .generateFailed:
            return "ZipVoice 沒有生成有效音訊。"
        case let .writeFailed(path):
            return "寫出生成音檔失敗：\(path)"
        }
    }
}

final class SherpaZipVoiceRuntime: @unchecked Sendable {
    private let tts: OpaquePointer
    private let paths: SherpaZipVoiceModelPaths
    private let promptSamples: [Float]
    private let promptSampleRate: Int

    init(paths: SherpaZipVoiceModelPaths) throws {
        self.paths = paths

        let promptWave = paths.promptAudio.withCString { pathPointer in
            SherpaOnnxReadWave(pathPointer)
        }
        guard let promptWave else {
            throw SherpaZipVoiceError.readPromptFailed(paths.promptAudio)
        }
        defer { SherpaOnnxFreeWave(promptWave) }

        let promptCount = Int(promptWave.pointee.num_samples)
        guard promptCount > 0, let promptPointer = promptWave.pointee.samples else {
            throw SherpaZipVoiceError.readPromptFailed(paths.promptAudio)
        }
        self.promptSamples = Array(UnsafeBufferPointer(start: promptPointer, count: promptCount))
        self.promptSampleRate = Int(promptWave.pointee.sample_rate)

        let cStrings = CStringBag()
        let zipvoice = SherpaOnnxOfflineTtsZipvoiceModelConfig(
            tokens: cStrings.add(paths.tokens),
            encoder: cStrings.add(paths.encoder),
            decoder: cStrings.add(paths.decoder),
            vocoder: cStrings.add(paths.vocoder),
            data_dir: cStrings.add(paths.dataDir),
            lexicon: cStrings.add(paths.lexicon),
            feat_scale: 0.1,
            t_shift: 0.5,
            target_rms: 0.1,
            guidance_scale: paths.guidanceScale
        )

        let model = SherpaOnnxOfflineTtsModelConfig(
            vits: SherpaOnnxOfflineTtsVitsModelConfig(),
            num_threads: 2,
            debug: 0,
            provider: cStrings.add("cpu"),
            matcha: SherpaOnnxOfflineTtsMatchaModelConfig(),
            kokoro: SherpaOnnxOfflineTtsKokoroModelConfig(),
            kitten: SherpaOnnxOfflineTtsKittenModelConfig(),
            zipvoice: zipvoice,
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
            throw SherpaZipVoiceError.createTtsFailed
        }
        self.tts = created
    }

    deinit {
        SherpaOnnxDestroyOfflineTts(tts)
    }

    func synthesize(text: String, outputURL: URL) throws {
        var cleanedSamples: [Float] = []
        let audio = text.withCString { textPointer in
            paths.promptText.withCString { promptTextPointer in
                "{\"min_char_in_sentence\":\"20\"}".withCString { extraPointer in
                    promptSamples.withUnsafeBufferPointer { promptBuffer -> UnsafePointer<SherpaOnnxGeneratedAudio>? in
                        var generation = SherpaOnnxGenerationConfig(
                            silence_scale: 0.2,
                            speed: paths.speed,
                            sid: 0,
                            reference_audio: promptBuffer.baseAddress,
                            reference_audio_len: Int32(promptBuffer.count),
                            reference_sample_rate: Int32(promptSampleRate),
                            reference_text: promptTextPointer,
                            num_steps: Int32(paths.numSteps),
                            extra: extraPointer
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
                }
            }
        }

        guard let audio else {
            throw SherpaZipVoiceError.generateFailed
        }
        defer { SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio) }

        let sampleCount = Int(audio.pointee.n)
        guard sampleCount > 0, let samples = audio.pointee.samples else {
            throw SherpaZipVoiceError.generateFailed
        }

        cleanedSamples = Self.mildClean(
            Array(UnsafeBufferPointer(start: samples, count: sampleCount)),
            sampleRate: Float(audio.pointee.sample_rate)
        )

        let writeResult = outputURL.path.withCString { outputPathPointer in
            cleanedSamples.withUnsafeBufferPointer { buffer in
                SherpaOnnxWriteWave(
                    buffer.baseAddress,
                    Int32(buffer.count),
                    audio.pointee.sample_rate,
                    outputPathPointer
                )
            }
        }

        guard writeResult == 1 else {
            throw SherpaZipVoiceError.writeFailed(outputURL.path)
        }
    }

    private static func mildClean(_ input: [Float], sampleRate: Float) -> [Float] {
        guard !input.isEmpty else { return input }

        let mean = input.reduce(Float(0), +) / Float(input.count)
        let dcRemoved = input.map { $0 - mean }

        let highPassed = onePoleHighPass(dcRemoved, cutoff: 65, sampleRate: sampleRate)
        var filtered = onePoleLowPass(highPassed, cutoff: 10_500, sampleRate: sampleRate)

        let limiterDrive: Float = 1.35
        filtered = filtered.map { sample in
            tanh(sample * limiterDrive) / limiterDrive
        }

        let rms = sqrt(filtered.reduce(Float(0)) { $0 + $1 * $1 } / Float(filtered.count))
        let targetRms = pow(Float(10), Float(-20) / 20)
        if rms > 0.000_001 {
            let gain = min(targetRms / rms, 3.5)
            filtered = filtered.map { $0 * gain }
        }

        let peakLimit = pow(Float(10), Float(-1) / 20)
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
