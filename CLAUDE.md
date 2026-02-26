# CLAUDE.md — PostureGuard

## Project Overview

PostureGuard is a macOS menu bar application that silently monitors your posture via webcam using MediaPipe pose/face detection. It runs invisibly (no dock icon, no window) and alerts you with voice feedback when it detects slouching or other posture issues. Detection is calibration-based — it learns the user's good posture rather than using generic thresholds.

## Tech Stack

- **Language:** Python 3.9+
- **Platform:** macOS 12+ only (uses `rumps` for menu bar, `say` for TTS)
- **Core dependencies:** mediapipe, opencv-python, rumps, numpy
- **Build tool:** py2app (for .app bundle creation)

## Project Structure

```
PostureGuard/
├── postureguard.py       # Main application — menu bar app, monitoring loop, calibration
├── camera_preview.py     # Standalone camera preview window (launched as subprocess)
├── setup.py              # py2app build configuration for macOS .app bundle
├── README.md             # User-facing documentation
└── .gitignore            # Ignores build/, dist/, __pycache__, .pyc, .DS_Store
```

This is a small, focused codebase (~660 lines of Python across 2 source files).

## Architecture

### Threading Model

```
Main Thread (rumps event loop)
├── Menu bar UI and user interactions
├── Spawns background monitoring thread
└── Spawns camera preview as separate subprocess

Background Thread (_monitor_loop)
├── Camera capture every 0.5s
├── MediaPipe pose + face detection
├── Metric extraction and baseline comparison
├── Menu bar title/item updates
└── Voice alert triggering

Subprocess (camera_preview.py)
└── Independent OpenCV window with HUD overlay
```

The camera preview runs as a subprocess (not a thread) because macOS requires OpenCV windows on the main thread, but `rumps` already owns the main thread.

### Key Design Patterns

- **Calibration-based detection:** `extract_metrics()` produces normalized landmark ratios. During calibration, these are averaged over ~45 frames and saved to `~/posture_calibration.json`. During monitoring, current metrics are compared against this baseline via `compare_to_baseline()`.
- **Score smoothing:** A rolling window of the last 20 scores (30 in preview) is averaged with `numpy.mean` to prevent UI flickering.
- **Severity scaling:** Each issue calculates severity as `min(1.0, deviation / (threshold * 3))` and applies a weighted penalty (head drop: 30, slouch: 35, lean: 20, shoulders: 20, forward: 25).
- **Voice alert cooldown:** Alerts trigger after 5 seconds of sustained bad posture, then enforce a 45-second cooldown between alerts.
- **Global threshold mutation:** Sensitivity presets modify global threshold variables directly.

### Shared Code Between Files

`extract_metrics()`, `compare_to_baseline()`, and `load_calibration()` are **duplicated** between `postureguard.py` and `camera_preview.py` (not shared via import). The camera_preview version includes slightly fewer metric keys (no `nose_y`, `mid_ear_y`, `mid_shoulder_y`) and shows diagnostic values in issue labels. Keep both copies in sync when modifying detection logic.

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `CALIBRATION_FILE` | `~/posture_calibration.json` | Both files |
| `CAMERA_INDEX` | `0` | Both files |
| `CHECK_INTERVAL` | `0.5` seconds | postureguard.py |
| `BAD_POSTURE_SECONDS` | `5` | postureguard.py |
| `COOLDOWN_SECONDS` | `45` | postureguard.py |
| Visibility threshold | `0.4` | Both files (in extract_metrics) |

### Detection Thresholds (Medium/Default)

| Threshold | Value |
|-----------|-------|
| `THRESH_HEAD_DROP` | 0.04 |
| `THRESH_SLOUCH` | 0.06 |
| `THRESH_HEAD_FORWARD` | 0.03 |
| `THRESH_SHOULDER_TILT` | 0.025 |

Sensitivity presets scale these: Low = ~1.5x, High = ~0.625x.

## Running the Application

```bash
# Direct execution
python3 postureguard.py

# Build macOS .app (alias mode for development)
python3 setup.py py2app -A

# Build standalone .app
python3 setup.py py2app
```

## Development Notes

### No Test Suite or CI/CD

This project currently has no automated tests, linting configuration, or CI/CD pipeline. Testing is done manually via the camera preview and menu bar interaction.

### Code Style

The codebase follows PEP 8 informally. No formatter or linter is configured. When making changes, match the existing style:
- 4-space indentation
- Double quotes for user-facing strings, single quotes for internal ones (inconsistent — match nearby code)
- Functions use snake_case
- Constants use UPPER_SNAKE_CASE
- Class follows rumps.App conventions

### Important Constraints

- **macOS only:** `rumps`, `say` command, and py2app are all macOS-specific. Do not introduce cross-platform abstractions unless explicitly requested.
- **Privacy-first:** Calibration stores landmark ratios only, never images or video. Do not change this.
- **No dock icon:** `LSUIElement: True` in setup.py keeps the app out of the Dock. The app should remain invisible except for the menu bar.
- **Camera preview is a subprocess:** Do not attempt to merge it into the main process. macOS OpenCV window requirements conflict with rumps.

### Common Modification Points

- **Adding new posture checks:** Add metric extraction in `extract_metrics()` (both files), add comparison logic in `compare_to_baseline()` (both files), and update calibration storage if new baseline keys are needed.
- **Changing thresholds:** Modify the `THRESH_*` constants and the three `set_sensitivity_*` methods in `PostureGuardApp`.
- **Menu bar items:** Modify `PostureGuardApp.__init__()` menu list and add corresponding callback methods.
- **Voice alerts:** Modify the `say()` function or the alert text in `_monitor_loop()`.

### Persistent State

The only persistent file is `~/posture_calibration.json`, created during calibration. It contains averaged metric values from the user's good posture capture session.
