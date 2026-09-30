// Draws the app icon and writes Resources/AppIcon.icns.
//
//   swift scripts/make-icon.swift
//
// An original design (SF Symbols and the Apple logo may not be used in app icons):
// a rounded square with an indigo-to-teal gradient, a white chat bubble, and a four-point sparkle.
import AppKit
import Foundation

let root = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().deletingLastPathComponent()
let iconset = FileManager.default.temporaryDirectory.appending(path: "AppIcon.iconset")
let output = root.appending(path: "Resources/AppIcon.icns")

let indigo = CGColor(red: 0.31, green: 0.27, blue: 0.90, alpha: 1)
let teal = CGColor(red: 0.05, green: 0.65, blue: 0.64, alpha: 1)

/// A four-point star: long points up/down/left/right, pinched towards the middle.
func sparkle(center c: CGPoint, radius r: CGFloat) -> CGPath {
    let path = CGMutablePath()
    let pinch = r * 0.18
    path.move(to: CGPoint(x: c.x, y: c.y + r))
    path.addQuadCurve(to: CGPoint(x: c.x + r, y: c.y), control: CGPoint(x: c.x + pinch, y: c.y + pinch))
    path.addQuadCurve(to: CGPoint(x: c.x, y: c.y - r), control: CGPoint(x: c.x + pinch, y: c.y - pinch))
    path.addQuadCurve(to: CGPoint(x: c.x - r, y: c.y), control: CGPoint(x: c.x - pinch, y: c.y - pinch))
    path.addQuadCurve(to: CGPoint(x: c.x, y: c.y + r), control: CGPoint(x: c.x - pinch, y: c.y + pinch))
    path.closeSubpath()
    return path
}

/// Draw the icon on a 1024 x 1024 canvas (origin bottom-left), scaled to `size`.
func render(size: Int) -> Data {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size, bitsPerSample: 8,
                               samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                               bytesPerRow: 0, bitsPerPixel: 0)!
    let context = NSGraphicsContext(bitmapImageRep: rep)!
    let cg = context.cgContext
    cg.scaleBy(x: CGFloat(size) / 1024, y: CGFloat(size) / 1024)
    let space = CGColorSpaceCreateDeviceRGB()

    // Body: macOS icon grid, 824 x 824 with a continuous-corner feel, on a soft shadow.
    let body = CGRect(x: 100, y: 100, width: 824, height: 824)
    let bodyPath = CGPath(roundedRect: body, cornerWidth: 186, cornerHeight: 186, transform: nil)
    cg.saveGState()
    cg.setShadow(offset: CGSize(width: 0, height: -10), blur: 28, color: CGColor(gray: 0, alpha: 0.28))
    cg.addPath(bodyPath)
    cg.setFillColor(indigo)
    cg.fillPath()
    cg.restoreGState()

    cg.saveGState()
    cg.addPath(bodyPath)
    cg.clip()
    let background = CGGradient(colorsSpace: space, colors: [indigo, teal] as CFArray, locations: [0, 1])!
    cg.drawLinearGradient(background, start: CGPoint(x: 150, y: 924), end: CGPoint(x: 874, y: 100), options: [])
    // A gentle highlight across the top half.
    let shine = CGGradient(colorsSpace: space, colors: [CGColor(gray: 1, alpha: 0.18), CGColor(gray: 1, alpha: 0)] as CFArray, locations: [0, 1])!
    cg.drawLinearGradient(shine, start: CGPoint(x: 512, y: 924), end: CGPoint(x: 512, y: 520), options: [])
    cg.restoreGState()

    // Chat bubble with a tail at the bottom left.
    let bubble = CGMutablePath()
    bubble.addRoundedRect(in: CGRect(x: 232, y: 330, width: 560, height: 400), cornerWidth: 130, cornerHeight: 130)
    bubble.move(to: CGPoint(x: 330, y: 350))
    bubble.addLine(to: CGPoint(x: 272, y: 250))
    bubble.addQuadCurve(to: CGPoint(x: 470, y: 336), control: CGPoint(x: 380, y: 300))
    bubble.closeSubpath()
    cg.saveGState()
    cg.setShadow(offset: CGSize(width: 0, height: -8), blur: 20, color: CGColor(gray: 0, alpha: 0.22))
    cg.addPath(bubble)
    cg.setFillColor(CGColor(gray: 1, alpha: 1))
    cg.fillPath()
    cg.restoreGState()

    // Sparkles inside the bubble, filled with the background gradient.
    let stars = CGMutablePath()
    stars.addPath(sparkle(center: CGPoint(x: 488, y: 522), radius: 128))
    stars.addPath(sparkle(center: CGPoint(x: 640, y: 628), radius: 52))
    stars.addPath(sparkle(center: CGPoint(x: 652, y: 432), radius: 34))
    cg.saveGState()
    cg.addPath(stars)
    cg.clip()
    cg.drawLinearGradient(background, start: CGPoint(x: 360, y: 660), end: CGPoint(x: 690, y: 390), options: [])
    cg.restoreGState()

    context.flushGraphics()
    return rep.representation(using: .png, properties: [:])!
}

try? FileManager.default.removeItem(at: iconset)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for base in [16, 32, 128, 256, 512] {
    try render(size: base).write(to: iconset.appending(path: "icon_\(base)x\(base).png"))
    try render(size: base * 2).write(to: iconset.appending(path: "icon_\(base)x\(base)@2x.png"))
}
try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
try render(size: 1024).write(to: root.appending(path: "Resources/AppIcon-1024.png"))

let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["-c", "icns", iconset.path, "-o", output.path]
try iconutil.run()
iconutil.waitUntilExit()
print(iconutil.terminationStatus == 0 ? "Wrote \(output.path)" : "iconutil failed")
