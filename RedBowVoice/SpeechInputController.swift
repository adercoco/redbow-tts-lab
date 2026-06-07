import AVFoundation
import Foundation
import Speech

@MainActor
final class SpeechInputController: ObservableObject {
    enum State: Equatable {
        case idle
        case authorizing
        case listening
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var partialText = ""

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh_TW"))
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var onFinalText: (@MainActor @Sendable (String) -> Void)?
    private var hasInputTap = false
    private var wantsToListen = false
    private var sessionID = 0

    var isListening: Bool {
        if case .listening = state { return true }
        return false
    }

    func toggle(onFinalText: @escaping @MainActor @Sendable (String) -> Void) {
        if isListening {
            stop()
            return
        }

        Task {
            await start(onFinalText: onFinalText)
        }
    }

    func start(onFinalText: @escaping @MainActor @Sendable (String) -> Void) async {
        guard !isListening else { return }

        sessionID += 1
        let activeSessionID = sessionID
        wantsToListen = true
        state = .authorizing
        partialText = ""

        do {
            if SFSpeechRecognizer.authorizationStatus() == .notDetermined {
                let speechOK = await requestSpeechAuthorization()
                guard wantsToListen, activeSessionID == sessionID else {
                    state = .idle
                    return
                }
                wantsToListen = false
                state = speechOK ? .idle : .failed("沒有語音辨識權限。")
                return
            }

            let speechOK = await requestSpeechAuthorization()
            guard wantsToListen, activeSessionID == sessionID else {
                state = .idle
                return
            }
            guard speechOK else {
                state = .failed("沒有語音辨識權限。")
                return
            }

            if AVAudioSession.sharedInstance().recordPermission == .undetermined {
                let micOK = await requestMicrophoneAuthorization()
                guard wantsToListen, activeSessionID == sessionID else {
                    state = .idle
                    return
                }
                wantsToListen = false
                state = micOK ? .idle : .failed("沒有麥克風權限。")
                return
            }

            let micOK = await requestMicrophoneAuthorization()
            guard wantsToListen, activeSessionID == sessionID else {
                state = .idle
                return
            }
            guard micOK else {
                state = .failed("沒有麥克風權限。")
                return
            }

            try startRecognition(sessionID: activeSessionID, onFinalText: onFinalText)
        } catch {
            cleanupRecognition()
            state = .failed(error.localizedDescription)
        }
    }

    func stop() {
        finish(submit: false)
    }

    func stopAndSubmit() {
        finish(submit: true)
    }

    private func finish(submit: Bool) {
        wantsToListen = false
        sessionID += 1
        let text = partialText.trimmingCharacters(in: .whitespacesAndNewlines)
        let handler = onFinalText

        cleanupRecognition()
        state = .idle

        if submit {
            if text.isEmpty {
                state = .failed("沒聽到清楚的中文，按住再說一次。")
            } else {
                handler?(text)
            }
        }
    }

    private func cleanupRecognition() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        if hasInputTap {
            audioEngine.inputNode.removeTap(onBus: 0)
            hasInputTap = false
        }
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
        onFinalText = nil

        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            // The next recording/speech session can recover by setting the audio session active again.
        }
    }

    private func startRecognition(sessionID activeSessionID: Int, onFinalText: @escaping @MainActor @Sendable (String) -> Void) throws {
        cleanupRecognition()
        guard wantsToListen, activeSessionID == sessionID else {
            state = .idle
            return
        }

        guard let recognizer, recognizer.isAvailable else {
            state = .failed("目前無法使用中文語音辨識。")
            return
        }

        self.onFinalText = onFinalText

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        self.request = request

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .spokenAudio, options: [.duckOthers, .defaultToSpeaker, .allowBluetooth])
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        let tapFormat: AVAudioFormat? = format.sampleRate > 0 && format.channelCount > 0 ? format : nil
        Self.installInputTap(on: input, request: request, format: tapFormat)
        hasInputTap = true

        audioEngine.prepare()
        try audioEngine.start()
        state = .listening

        task = recognizer.recognitionTask(
            with: request,
            resultHandler: Self.makeRecognitionHandler(
                controller: self,
                sessionID: activeSessionID,
                onFinalText: onFinalText
            )
        )
    }

    nonisolated private static func installInputTap(
        on input: AVAudioInputNode,
        request: SFSpeechAudioBufferRecognitionRequest,
        format: AVAudioFormat?
    ) {
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak request] buffer, _ in
            request?.append(buffer)
        }
    }

    nonisolated private static func makeRecognitionHandler(
        controller: SpeechInputController,
        sessionID activeSessionID: Int,
        onFinalText: @escaping @MainActor @Sendable (String) -> Void
    ) -> (SFSpeechRecognitionResult?, Error?) -> Void {
        { [weak controller] result, error in
            let text = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            let errorMessage = error?.localizedDescription

            Task { @MainActor in
                guard let controller else { return }
                guard controller.request != nil, controller.sessionID == activeSessionID else { return }
                if let text {
                    controller.partialText = text
                    if isFinal {
                        controller.finish(submit: false)
                        onFinalText(text)
                    }
                }

                if let errorMessage {
                    controller.finish(submit: false)
                    controller.state = .failed(errorMessage)
                }
            }
        }
    }

    nonisolated private func requestSpeechAuthorization() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                DispatchQueue.main.async {
                    continuation.resume(returning: status == .authorized)
                }
            }
        }
    }

    nonisolated private func requestMicrophoneAuthorization() async -> Bool {
        await withCheckedContinuation { continuation in
            if #available(iOS 17.0, *) {
                AVAudioApplication.requestRecordPermission { granted in
                    DispatchQueue.main.async {
                        continuation.resume(returning: granted)
                    }
                }
            } else {
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    DispatchQueue.main.async {
                        continuation.resume(returning: granted)
                    }
                }
            }
        }
    }
}
