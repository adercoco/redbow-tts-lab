import SwiftUI

struct VoicePreset: Identifiable, Equatable {
    let id: String
    let name: String
    let subtitle: String
    let description: String
    let modelHint: String
    let modelPackDirectory: String?
    let defaultPrompt: String
    let tint: Color
    let previewAudioPath: String?
    let isPreviewOnly: Bool

    init(
        id: String,
        name: String,
        subtitle: String,
        description: String,
        modelHint: String,
        modelPackDirectory: String?,
        defaultPrompt: String,
        tint: Color,
        previewAudioPath: String? = nil,
        isPreviewOnly: Bool = false
    ) {
        self.id = id
        self.name = name
        self.subtitle = subtitle
        self.description = description
        self.modelHint = modelHint
        self.modelPackDirectory = modelPackDirectory
        self.defaultPrompt = defaultPrompt
        self.tint = tint
        self.previewAudioPath = previewAudioPath
        self.isPreviewOnly = isPreviewOnly
    }

    var canPlayPreview: Bool {
        previewAudioPath != nil
    }

    static let all: [VoicePreset] = [
        .init(
            id: "cosy-zipvoice-4step-qvocoder",
            name: "Cosy 女聲 ZipVoice",
            subtitle: "4-step、MatMul qint8 Vocos、mild clean",
            description: "手機主候選：CosyVoice2 golden teacher 蒸餾到 ZipVoice student，4-step 推論，搭配 iOS 可執行的 MatMul-only qint8 Vocos 與 mild clean。QDQ-all int8 已下架，因為同文檢測出明顯高頻電子雜音。",
            modelHint: "provider: sherpa-onnx zipvoice int8; teacher: cosyvoice2 golden young Taiwanese female, low retroflex, soft endings; steps: 4; vocoder: matmul qint8 vocos; postprocess: mild clean",
            modelPackDirectory: "CosyZipVoice4StepQvocoder",
            defaultPrompt: "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
            tint: .pink
        ),
        .init(
            id: "matcha-cosy-golden",
            name: "Cosy 女聲 Matcha",
            subtitle: "30k / 16-step、HiFi-GAN、真離線生成",
            description: "用 CosyVoice2 golden teacher 語料 fine-tune 到 30k 的 Matcha 小模型。手機端走 sherpa-onnx Matcha acoustic ONNX 16-step + HiFi-GAN vocoder，中文先離線轉 compact pinyin。",
            modelHint: "provider: sherpa-onnx matcha; teacher: cosyvoice2 golden young Taiwanese female, low retroflex, soft endings; frontend: zh text to compact pinyin private-use lexicon; checkpoint: 30k; steps: 16; vocoder: HiFi-GAN",
            modelPackDirectory: "MatchaCosyGolden",
            defaultPrompt: "你先不要急，我们慢慢来，把事情一件一件处理好。",
            tint: .pink
        ),
        .init(
            id: "taiwan-mandarin-low-r",
            name: "台大低卷舌女聲",
            subtitle: "台灣國語、清亮溫柔、克制可愛",
            description: "目標聲音仍是 CosyVoice2 golden teacher 的台灣低卷舌女聲；目前 tiny Cosy 蒸餾版 ASR gate failed，所以 app 先 fallback 到既有 Qwen→ZipVoice 包。",
            modelHint: "provider: sherpa-onnx zipvoice int8; voice: qwen_zipvoice_fallback_until_cosy_passes_asr; zh Mandarin with Taiwan accent, low retroflex, soft endings",
            modelPackDirectory: "QwenZipVoiceDistillInt8",
            defaultPrompt: "你先不要急，我们慢慢来，把事情一件一件处理好。",
            tint: .pink
        ),
        .init(
            id: "bow-detective",
            name: "紅領結偵探",
            subtitle: "明亮、機靈、少年感",
            description: "第一個要測的核心聲線：少年、清楚、反應快。之後接 cloning model 時用同一個 preset id。",
            modelHint: "target: energetic boy detective, zh-TW/zh-CN Mandarin, crisp diction",
            modelPackDirectory: nil,
            defaultPrompt: "真相只有一個，現在輪到我來推理了。",
            tint: .red
        ),
        .init(
            id: "taiwan-variety",
            name: "台灣紅人",
            subtitle: "綜藝、親切、有記憶點",
            description: "用來測台灣口吻和情緒表現。正式版應使用授權聲線或原創角色聲音。",
            modelHint: "target: Taiwanese variety host energy, zh-TW Mandarin, warm and expressive",
            modelPackDirectory: nil,
            defaultPrompt: "各位觀眾朋友，今天這個真的太有意思了。",
            tint: .orange
        ),
        .init(
            id: "news-anchor",
            name: "新聞主播",
            subtitle: "標準、穩、清楚",
            description: "拿來當中文清晰度基準：咬字、停頓、長句穩定度。",
            modelHint: "target: Taiwanese news anchor, zh-TW Mandarin, neutral and stable",
            modelPackDirectory: nil,
            defaultPrompt: "根據最新消息，這項測試將用來比較不同聲音模型的品質。",
            tint: .cyan
        ),
        .init(
            id: "warm-narrator",
            name: "溫柔旁白",
            subtitle: "自然、舒服、耐聽",
            description: "用來測長句和自然度，避免只在短台詞好聽。",
            modelHint: "target: warm young narrator, zh-TW Mandarin, natural pacing",
            modelPackDirectory: nil,
            defaultPrompt: "如果聲音聽起來自然，我們就可以繼續往角色化方向調整。",
            tint: .pink
        ),
        .init(
            id: "dramatic-reveal",
            name: "揭曉大叔",
            subtitle: "低沉、戲劇、懸疑感",
            description: "用來測低音男性聲線和情緒誇張時會不會破音。",
            modelHint: "target: dramatic middle-aged reveal voice, Mandarin, suspenseful",
            modelPackDirectory: nil,
            defaultPrompt: "答案其實一直都在我們眼前，只是沒有人注意到。",
            tint: .brown
        )
    ]

    static let appModels: [VoicePreset] = [
        all[0]
    ]
}
