import AppKit

/// The menu-bar version of the app icon: the same chat bubble and sparkles, in one colour,
/// as a template image so macOS tints it for light and dark menu bars.
enum MenuBarIcon {
    /// A solid bubble with the sparkles cut out; dimmed when the server is stopped.
    static func image(running: Bool) -> NSImage {
        let size = NSSize(width: 20, height: 18)
        let image = NSImage(size: size, flipped: false) { rect in
            guard let cg = NSGraphicsContext.current?.cgContext else { return false }
            // The app icon's bubble spans x 232...792 and y 250...730 on its 1024 canvas; fit that into the menu bar.
            let scale = min((rect.width - 1) / 560, (rect.height - 1) / 480)
            var transform = CGAffineTransform(translationX: (rect.width - 560 * scale) / 2, y: (rect.height - 480 * scale) / 2)
                .scaledBy(x: scale, y: scale)
                .translatedBy(x: -232, y: -250)
            guard let bubble = bubblePath().copy(using: &transform),
                  let stars = sparklesPath().copy(using: &transform) else { return false }

            // Stopped: the same icon, dimmed, the usual macOS look for something inactive.
            cg.setAlpha(running ? 1 : 0.4)
            cg.setFillColor(.black)
            cg.addPath(bubble)
            cg.fillPath()
            cg.setBlendMode(.clear)
            cg.addPath(stars)
            cg.fillPath()
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = running ? "Foundation Model Server, running" : "Foundation Model Server, stopped"
        return image
    }

    /// Same shapes as scripts/make-icon.swift, in its 1024 x 1024 coordinates.
    static func bubblePath() -> CGPath {
        let bubble = CGMutablePath()
        bubble.addRoundedRect(in: CGRect(x: 232, y: 330, width: 560, height: 400), cornerWidth: 130, cornerHeight: 130)
        bubble.move(to: CGPoint(x: 330, y: 350))
        bubble.addLine(to: CGPoint(x: 272, y: 250))
        bubble.addQuadCurve(to: CGPoint(x: 470, y: 336), control: CGPoint(x: 380, y: 300))
        bubble.closeSubpath()
        return bubble
    }

    static func sparklesPath() -> CGPath {
        let stars = CGMutablePath()
        stars.addPath(sparkle(center: CGPoint(x: 488, y: 522), radius: 150))
        stars.addPath(sparkle(center: CGPoint(x: 660, y: 630), radius: 64))
        return stars
    }

    private static func sparkle(center c: CGPoint, radius r: CGFloat) -> CGPath {
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
}
