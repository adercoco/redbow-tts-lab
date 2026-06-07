import Foundation

enum PinyinFrontendError: LocalizedError {
    case missingMap
    case emptyPinyin(String)

    var errorDescription: String? {
        switch self {
        case .missingMap:
            return "找不到離線 pinyin_map.json。"
        case let .emptyPinyin(text):
            return "這句話轉不成 Matcha pinyin：\(text)"
        }
    }
}

final class PinyinFrontend: @unchecked Sendable {
    static let shared = try? PinyinFrontend()

    private let map: [String: String]
    private let asciiToPrivateUseBase = 0xE000

    init(bundle: Bundle = .main) throws {
        guard let url = bundle.url(
            forResource: "pinyin_map",
            withExtension: "json",
            subdirectory: "RedBowAssets"
        ) else {
            throw PinyinFrontendError.missingMap
        }

        let data = try Data(contentsOf: url)
        self.map = try JSONDecoder().decode([String: String].self, from: data)
    }

    func encodeForMatcha(_ text: String) throws -> String {
        var compactPinyin = ""

        for scalar in text.unicodeScalars {
            let character = String(Character(scalar))
            if let mapped = map[character] {
                compactPinyin += mapped
                continue
            }

            switch scalar {
            case "，", "、", ",":
                compactPinyin += ","
            case "。", ".":
                compactPinyin += "."
            case "！", "!":
                compactPinyin += "!"
            case "？", "?":
                compactPinyin += "?"
            case "：", ":":
                compactPinyin += ":"
            case "；", ";":
                compactPinyin += ";"
            default:
                if scalar.properties.isWhitespace {
                    continue
                }
                if scalar.isASCII {
                    let value = CharacterSet.alphanumerics.contains(scalar) ? Character(scalar).lowercased() : ""
                    compactPinyin += value
                }
            }
        }

        let encoded = compactPinyin.compactMap { char -> String? in
            guard let scalar = char.unicodeScalars.first else { return nil }
            if scalar.isASCII, scalar.properties.isAlphabetic || CharacterSet.decimalDigits.contains(scalar) {
                return UnicodeScalar(asciiToPrivateUseBase + Int(scalar.value)).map { String(Character($0)) }
            }
            if ",.!?:;".contains(char) {
                return String(char)
            }
            return nil
        }.joined()

        guard !encoded.isEmpty else {
            throw PinyinFrontendError.emptyPinyin(text)
        }

        return encoded
    }
}
