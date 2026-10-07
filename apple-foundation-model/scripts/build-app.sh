#!/bin/bash
# Build "Foundation Model Server.app" into build/.
#
#   scripts/build-app.sh             # build and sign
#   scripts/build-app.sh --install   # also copy it to ~/Applications
#   scripts/build-app.sh --dist      # also zip it into dist/FoundationModelServer.zip, the copy kept in git
#
# Signs with your "Apple Development" certificate if you have one (set SIGN_IDENTITY to choose another),
# otherwise ad hoc, which is enough for the on-device model.
#
# Private Cloud Compute: once Apple has granted your team the entitlement, download a macOS provisioning
# profile that includes it, then build with the profile and the bundle id it was made for:
#   PROVISIONING_PROFILE=~/Downloads/FoundationModelServer.provisionprofile \
#   BUNDLE_ID=com.yourname.FoundationModelServer scripts/build-app.sh --install
set -euo pipefail
cd "$(dirname "$0")/.."

APP_NAME="Foundation Model Server"
BUNDLE_ID="${BUNDLE_ID:-local.foundationmodelserver}"
VERSION="1.0"
APP="build/$APP_NAME.app"

swift build -c release --arch arm64

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp .build/release/FoundationModelServer "$APP/Contents/MacOS/FoundationModelServer"
# The app icon. Regenerate it with: swift scripts/make-icon.swift
cp Resources/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleExecutable</key><string>FoundationModelServer</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>27.0</string>
  <!-- Menu-bar app: no Dock icon. -->
  <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

# With a provisioning profile, embed it and sign with exactly the entitlements it grants.
ENTITLEMENTS=()
if [ -n "${PROVISIONING_PROFILE:-}" ]; then
  cp "$PROVISIONING_PROFILE" "$APP/Contents/embedded.provisionprofile"
  security cms -D -i "$PROVISIONING_PROFILE" > build/profile.plist
  /usr/libexec/PlistBuddy -x -c "Print :Entitlements" build/profile.plist > build/entitlements.plist
  if ! grep -q "com.apple.developer.private-cloud-compute" build/entitlements.plist; then
    echo "Warning: this profile doesn't include com.apple.developer.private-cloud-compute." >&2
  fi
  ENTITLEMENTS=(--entitlements build/entitlements.plist)
fi

IDENTITY="${SIGN_IDENTITY:-$(security find-identity -v -p codesigning 2>/dev/null | grep -m1 '"Apple Development' | awk '{print $2}' || true)}"
if [ -n "$IDENTITY" ]; then
  codesign --force --options runtime ${ENTITLEMENTS[@]+"${ENTITLEMENTS[@]}"} --sign "$IDENTITY" "$APP"
  echo "Signed with $(security find-identity -v -p codesigning | grep "$IDENTITY" | sed 's/.*"\(.*\)"/\1/')"
else
  codesign --force --sign - "$APP"
  echo "Signed ad hoc (no Apple Development certificate found)"
fi

if [ "${1:-}" = "--install" ]; then
  mkdir -p "$HOME/Applications"
  rm -rf "$HOME/Applications/$APP_NAME.app"
  cp -R "$APP" "$HOME/Applications/"
  echo "Installed to ~/Applications/$APP_NAME.app"
elif [ "${1:-}" = "--dist" ]; then
  mkdir -p dist
  rm -f dist/FoundationModelServer.zip
  ditto -c -k --keepParent "$APP" dist/FoundationModelServer.zip
  echo "Wrote dist/FoundationModelServer.zip"
else
  echo "Built $APP"
fi
