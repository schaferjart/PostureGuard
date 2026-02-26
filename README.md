# PostureGuard

An invisible macOS menu bar app that monitors your posture through your webcam and yells at you when you slouch.

No window. No dock icon. Just silent, constant judgment — and a voice that tells you to sit up straight.

## How It Works

PostureGuard uses [MediaPipe](https://github.com/google/mediapipe) Pose and FaceMesh to track your head, face, and shoulders in real time via your webcam. On first launch, you calibrate it by sitting in your best posture. From then on, it measures deviations from **your** baseline — not generic thresholds — and alerts you when you start slouching.

### What It Detects

- **Head dropping** — chin drifting down (tech neck)
- **Slouching** — your ear-to-shoulder distance shrinking as you compress
- **Leaning left/right** — head drifting off-center from your shoulders
- **Shoulder tilt** — one shoulder higher than the other
- **Leaning forward** — face getting closer to the camera

### How It Alerts You

- Menu bar icon changes: `PG` (good) → `PG!` (warning) → `PG!!` (bad)
- macOS notification with your current score and the issue
- Voice alert via text-to-speech (Samantha) after 5 seconds of sustained bad posture
- 45-second cooldown between voice alerts so it doesn't drive you insane

## Requirements

- macOS 12+ (Monterey or later)
- Python 3.9+
- A built-in or external webcam
- Camera permissions granted to the terminal/app

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/schaferjart/PostureGuard.git
cd PostureGuard
```

### 2. Install dependencies

```bash
pip3 install opencv-python mediapipe numpy rumps
```

If `mediapipe` tries to build `opencv-contrib-python` from source (takes forever), install without dependency resolution and add the missing pieces:

```bash
pip3 install opencv-python mediapipe --no-deps
pip3 install absl-py protobuf sounddevice numpy matplotlib
```

**NumPy compatibility note:** If you get `numpy.core.multiarray` import errors, your matplotlib may be too old for your numpy version. Fix with:

```bash
pip3 install --upgrade matplotlib
```

### 3. Run it

```bash
python3 postureguard.py
```

Look for **PG** in your menu bar. Click it to see the menu.

### 4. (Optional) Build as a macOS .app

This gives you a proper app bundle with no dock icon and camera permission prompts:

```bash
pip3 install py2app
python3 setup.py py2app -A
open dist/PostureGuard.app
```

The `-A` flag creates an alias build (fast, for development). For a standalone redistributable build:

```bash
python3 setup.py py2app
```

### 5. (Optional) Auto-start on login

```bash
osascript -e 'tell application "System Events" to make login item at end with properties {path:"/path/to/PostureGuard/dist/PostureGuard.app", hidden:true}'
```

Replace `/path/to/PostureGuard` with your actual path.

## Usage

### First Launch

1. Click **PG** in the menu bar
2. Click **Calibrate Good Posture**
3. Sit up straight — best posture you've got
4. Hold still for ~3 seconds while it captures your baseline
5. You'll hear "Calibration complete. I'm watching you."
6. Monitoring starts automatically

Your calibration is saved to `~/posture_calibration.json` and persists across sessions.

### Menu Bar Options

| Menu Item | What It Does |
|---|---|
| **Status** | Shows if monitoring is active or paused |
| **Score** | Your current posture score (0-100%) |
| **Issues** | The most pressing posture issue right now |
| **Stop/Start Monitoring** | Toggle posture tracking on/off |
| **Show Camera** | Opens a live camera window with skeleton overlay, face mesh, and score HUD — useful for seeing exactly what the app sees and debugging |
| **Calibrate Good Posture** | Recapture your baseline (do this when you change chairs, camera angle, etc.) |
| **Sensitivity** | Low (relaxed) / Medium / High (strict) |
| **Quit PostureGuard** | Stops monitoring and exits |

### Camera Preview

Click **Show Camera** to open a live view showing:
- Green skeleton overlay on your body
- Cyan face mesh contours
- Posture score bar (green/yellow/red)
- Issue labels in red when posture is bad
- Flashing red border when things are really bad

Press `q` in the camera window or click **Hide Camera** to close it.

### Sensitivity Levels

| Level | Description | Best For |
|---|---|---|
| **Low** | Relaxed thresholds, only flags major slouching | Casual use, couch computing |
| **Medium** | Balanced detection (default) | Regular desk work |
| **High** | Strict, catches subtle deviations | When you're serious about posture |

## Files

| File | Purpose |
|---|---|
| `postureguard.py` | Main menu bar app — background monitoring, alerts, calibration |
| `camera_preview.py` | Standalone camera window with live skeleton/score overlay |
| `setup.py` | py2app config to build as a macOS .app bundle |
| `~/posture_calibration.json` | Your calibration data (auto-generated, not in repo) |

## Architecture

```
postureguard.py (menu bar app, rumps)
├── Background thread: captures frames, runs MediaPipe, computes score
├── Main thread: rumps event loop, menu bar updates
└── Subprocess: camera_preview.py (launched on demand)

camera_preview.py (standalone OpenCV window)
├── Independent camera capture + MediaPipe processing
├── Reads calibration from ~/posture_calibration.json
└── Displays live HUD with skeleton overlay
```

The menu bar app and camera preview run as separate processes because macOS requires OpenCV windows (`cv2.imshow`) to run on the main thread, but `rumps` already owns the main thread for the menu bar event loop.

## Camera Permissions

macOS requires explicit camera access. On first run:

1. You may see a "Python wants to access the camera" dialog — click **Allow**
2. If no dialog appears, go to **System Preferences → Privacy & Security → Camera** and enable access for your terminal app (Terminal, iTerm2, etc.) or PostureGuard.app

## Troubleshooting

**"Score: --" and nothing happens**
- You need to calibrate first. Click PG → Calibrate Good Posture.

**Camera not working**
- Check System Preferences → Privacy & Security → Camera
- Make sure your terminal app or PostureGuard.app has camera permission
- If you recently changed permissions, restart the app

**Score always 100% / never triggers**
- Your calibration might be stale. Recalibrate with your current setup.
- Try **High** sensitivity from the Sensitivity menu.
- Make sure the camera can see your face AND shoulders.

**Too many false alarms**
- Switch to **Low** sensitivity
- Recalibrate in a more natural (but still upright) position — don't sit unnaturally straight during calibration or everything will seem like slouching.

**High CPU usage**
- The app checks posture every 0.5 seconds (not every frame). Typical usage is ~15-25% of one CPU core. The camera preview uses more when open. Close the camera preview when you don't need it.

## Privacy

All processing happens locally on your Mac. No images, video, or posture data are sent anywhere. The only file created is `~/posture_calibration.json`, which contains abstract landmark ratios — not images.

## License

MIT

## Credits

Built with:
- [MediaPipe](https://github.com/google/mediapipe) — pose and face landmark detection
- [OpenCV](https://opencv.org/) — camera capture and display
- [rumps](https://github.com/jaredks/rumps) — macOS menu bar framework
- [py2app](https://py2app.readthedocs.io/) — macOS app bundling
