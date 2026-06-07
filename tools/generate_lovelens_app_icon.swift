import AppKit
import Foundation

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let output = root
    .appendingPathComponent("LoveLensDemo/Assets.xcassets/AppIcon.appiconset", isDirectory: true)

try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

let iconSpecs: [(String, Int)] = [
    ("Icon-20@2x.png", 40),
    ("Icon-20@3x.png", 60),
    ("Icon-29@2x.png", 58),
    ("Icon-29@3x.png", 87),
    ("Icon-40@2x.png", 80),
    ("Icon-40@3x.png", 120),
    ("Icon-60@2x.png", 120),
    ("Icon-60@3x.png", 180),
    ("Icon-76@1x.png", 76),
    ("Icon-76@2x.png", 152),
    ("Icon-83.5@2x.png", 167),
    ("Icon-1024.png", 1024)
]

func drawIcon(size: Int) -> NSImage {
    let image = NSImage(size: NSSize(width: size, height: size))
    image.lockFocus()
    defer { image.unlockFocus() }

    let rect = NSRect(x: 0, y: 0, width: size, height: size)
    NSColor(red: 0.95, green: 0.28, blue: 0.43, alpha: 1).setFill()
    NSBezierPath(roundedRect: rect, xRadius: CGFloat(size) * 0.23, yRadius: CGFloat(size) * 0.23).fill()

    let bubbleRect = NSRect(
        x: CGFloat(size) * 0.24,
        y: CGFloat(size) * 0.38,
        width: CGFloat(size) * 0.38,
        height: CGFloat(size) * 0.25
    )
    NSColor.white.setFill()
    NSBezierPath(roundedRect: bubbleRect, xRadius: CGFloat(size) * 0.11, yRadius: CGFloat(size) * 0.11).fill()

    let tail = NSBezierPath()
    tail.move(to: NSPoint(x: bubbleRect.minX + bubbleRect.width * 0.16, y: bubbleRect.minY + CGFloat(size) * 0.03))
    tail.line(to: NSPoint(x: bubbleRect.minX + bubbleRect.width * 0.32, y: bubbleRect.minY + CGFloat(size) * 0.01))
    tail.line(to: NSPoint(x: bubbleRect.minX + bubbleRect.width * 0.25, y: bubbleRect.minY + CGFloat(size) * 0.10))
    tail.close()
    tail.fill()

    NSColor.white.withAlphaComponent(0.96).setStroke()
    let lensRect = NSRect(
        x: CGFloat(size) * 0.48,
        y: CGFloat(size) * 0.31,
        width: CGFloat(size) * 0.26,
        height: CGFloat(size) * 0.26
    )
    let lens = NSBezierPath(ovalIn: lensRect)
    lens.lineWidth = max(4, CGFloat(size) * 0.055)
    lens.stroke()

    let handle = NSBezierPath()
    handle.lineWidth = max(4, CGFloat(size) * 0.06)
    handle.lineCapStyle = .round
    handle.move(to: NSPoint(x: lensRect.maxX - CGFloat(size) * 0.02, y: lensRect.minY + CGFloat(size) * 0.03))
    handle.line(to: NSPoint(x: lensRect.maxX + CGFloat(size) * 0.12, y: lensRect.minY - CGFloat(size) * 0.11))
    handle.stroke()

    NSColor(red: 0.60, green: 0.24, blue: 0.33, alpha: 0.45).setFill()
    let lineHeight = CGFloat(size) * 0.035
    let line1 = NSRect(x: CGFloat(size) * 0.41, y: CGFloat(size) * 0.72, width: CGFloat(size) * 0.18, height: lineHeight)
    let line2 = NSRect(x: CGFloat(size) * 0.46, y: CGFloat(size) * 0.65, width: CGFloat(size) * 0.11, height: lineHeight)
    NSBezierPath(roundedRect: line1, xRadius: lineHeight / 2, yRadius: lineHeight / 2).fill()
    NSBezierPath(roundedRect: line2, xRadius: lineHeight / 2, yRadius: lineHeight / 2).fill()

    return image
}

func writePNG(_ image: NSImage, to url: URL) throws {
    guard let tiff = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let png = bitmap.representation(using: .png, properties: [:]) else {
        throw CocoaError(.fileWriteUnknown)
    }
    try png.write(to: url)
}

for (filename, size) in iconSpecs {
    try writePNG(drawIcon(size: size), to: output.appendingPathComponent(filename))
}
