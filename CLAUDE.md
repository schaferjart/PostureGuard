# CLAUDE.md — PostureGuard

## Project Overview

PostureGuard is a macOS menu bar application that silently monitors your posture via webcam using MediaPipe pose/face detection. It runs invisibly (no dock icon, no window) and alerts you with voice feedback when it detects slouching or other posture issues. Detection is calibration-based — it learns the user's good posture rather than using generic thresholds.

## Tech Stack

- **Language:** Python 3.9+
- **Platform:** macOS 12+ only (uses `rumps` for menu bar, `say` for TTS)
- **Core dependencies:** mediapipe, opencv-python, rumps, numpy (see `requirements.txt`)
- **Build tool:** py2app (for .app bundle creation)
- **Tests:** unittest (no external test framework required)

## Project Structure

```
PostureGuard/
├── posture_core.py       # Shared detection logic, constants, calibration I/O
├── postureguard.py       # Main application — menu bar app, monitoring loop, calibration
├── camera_preview.py     # Standalone camera preview window (launched as subprocess)
├── test_posture_core.py  # Unit tests for core detection logic (23 tests)
├── setup.py              # py2app build configuration for macOS .app bundle
├── requirements.txt      # Python dependency pins
├── README.md             # User-facing documentation
├── CLAUDE.md             # This file — AI assistant context
└── .gitignore            # Ignores build/, dist/, __pycache__, .pyc, .DS_Store
```

## Architecture

### Module Dependency Graph

```
posture_core.py          ← shared constants, detection logic, calibration I/O
├── postureguard.py      ← imports from posture_core
├── camera_preview.py    ← imports from posture_core
└── test_posture_core.py ← imports from posture_core (pure-logic functions only)
```

`posture_core.py` is the single source of truth for detection logic. Both `postureguard.py` and `camera_preview.py` import from it — there is no duplicated code.

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
├── Voice alert triggering
└── Posture session logging (every ~3s)

Subprocess (camera_preview.py)
└── Independent OpenCV window with HUD overlay
```

The camera preview runs as a subprocess (not a thread) because macOS requires OpenCV windows on the main thread, but `rumps` already owns the main thread.

### Key Design Patterns

- **Calibration-based detection:** `extract_metrics()` produces normalized landmark ratios. During calibration, these are averaged via `average_metrics()` and saved to `~/posture_calibration.json`. During monitoring, current metrics are compared against this baseline via `compare_to_baseline()`.
- **Threshold parameterization:** `compare_to_baseline()` accepts an optional `thresholds` dict, making sensitivity switching clean — no global mutation needed in the core logic. Sensitivity presets live in `SENSITIVITY_PRESETS` dict.
- **Score smoothing:** `smooth_score()` maintains a rolling window (20 readings in monitor, 30 in preview) averaged with `numpy.mean` to prevent UI flickering.
- **Severity scaling:** Each issue calculates severity as `min(1.0, deviation / (threshold * 3))` and applies a weighted penalty (head drop: 30, slouch: 35, lean: 20, shoulders: 20, forward: 25).
- **Structured issues:** `compare_to_baseline()` returns issues as `list[tuple[str, float]]` — each issue is `(message, deviation_value)`, enabling both user-facing labels and diagnostic overlays.
- **Voice alert cooldown:** Alerts trigger after 5 seconds of sustained bad posture, then enforce a 45-second cooldown between alerts.
- **Lazy MediaPipe loading:** `_init_mediapipe()` defers the MediaPipe import so pure-logic functions (comparison, calibration I/O, smoothing) can be tested without MediaPipe installed.
- **Session logging:** Posture scores are logged to `~/PostureGuard/posture_log.csv` every ~3 seconds. A "Today's Summary" menu item shows daily stats (average score, good posture %, top issue).

## Key Constants (in posture_core.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| `CALIBRATION_FILE` | `~/posture_calibration.json` | Baseline storage |
| `CAMERA_INDEX` | `0` | Default webcam |
| `CHECK_INTERVAL` | `0.5` seconds | Time between posture checks |
| `BAD_POSTURE_SECONDS` | `5` | Seconds before voice alert |
| `COOLDOWN_SECONDS` | `45` | Seconds between voice alerts |
| `MIN_VISIBILITY` | `0.4` | MediaPipe landmark visibility threshold |

### Detection Thresholds (Medium/Default)

| Threshold | Value |
|-----------|-------|
| `THRESH_HEAD_DROP` | 0.04 |
| `THRESH_SLOUCH` | 0.06 |
| `THRESH_HEAD_FORWARD` | 0.03 |
| `THRESH_SHOULDER_TILT` | 0.025 |

Presets in `SENSITIVITY_PRESETS`: Low = ~1.5x, Medium = default, High = ~0.625x.

## Running & Testing

```bash
# Run the app directly
python3 postureguard.py

# Run the test suite (23 tests, no mediapipe/camera required)
python3 -m unittest test_posture_core -v

# Build macOS .app (alias mode for development)
python3 setup.py py2app -A

# Build standalone .app
python3 setup.py py2app

# Install dependencies
pip3 install -r requirements.txt
```

## Development Notes

### Test Suite

`test_posture_core.py` contains 23 unit tests covering:
- **Baseline comparison:** All 5 posture issue types detected correctly
- **Scoring:** Perfect posture = 100, compound penalties, score floor at 0
- **Sensitivity:** Custom thresholds override defaults, low < medium < high strictness
- **Calibration I/O:** Save/load roundtrip, missing file returns None
- **Metric averaging:** Single/multi-frame averaging, output types
- **Score smoothing:** Averaging, max history length, integer output

Tests only exercise pure logic functions and do **not** require MediaPipe, OpenCV, or a camera. They run anywhere Python 3.9+ and numpy are available.

### Code Style

- 4-space indentation
- Functions use snake_case, constants use UPPER_SNAKE_CASE
- Match nearby quoting style (double quotes for user-facing strings)
- No formatter or linter is configured — match existing patterns

### Important Constraints

- **macOS only:** `rumps`, `say` command, and py2app are all macOS-specific. Do not introduce cross-platform abstractions unless explicitly requested.
- **Privacy-first:** Calibration stores landmark ratios only, never images or video. Session logs store scores and issue labels only. Do not change this.
- **No dock icon:** `LSUIElement: True` in setup.py keeps the app out of the Dock.
- **Camera preview is a subprocess:** Do not merge it into the main process. macOS OpenCV window requirements conflict with rumps.
- **Single source of truth:** All detection logic lives in `posture_core.py`. Do not duplicate logic between files.

### Common Modification Points

- **Adding new posture checks:** Add metric extraction in `extract_metrics()` (posture_core.py), add comparison logic in `compare_to_baseline()` (posture_core.py), add tests in `test_posture_core.py`, and update calibration storage if new baseline keys are needed.
- **Changing thresholds:** Modify the `THRESH_*` constants and `SENSITIVITY_PRESETS` dict in `posture_core.py`.
- **Menu bar items:** Modify `PostureGuardApp.__init__()` menu list and add corresponding callback methods in `postureguard.py`.
- **Voice alerts:** Modify the `say()` function or the alert text in `_monitor_loop()` in `postureguard.py`.

### Persistent State

| File | Location | Contents |
|------|----------|----------|
| `posture_calibration.json` | `~/` | Averaged metric values from good posture calibration |
| `posture_log.csv` | `~/PostureGuard/` | Timestamped posture scores and issue labels |
