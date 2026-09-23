# DiskChat Android APK

WebView shell around the DiskChat chat UI. Opens `.gguf` via intent filters.

## Build
```bash
cd android
./gradlew assembleDebug
# APK: app/build/outputs/apk/debug/app-debug.apk
```

Backend (optional, local): run `python diskchat.py --serve --port 8765` on device/Termux, or point UI at your server.
