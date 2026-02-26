#!/usr/bin/env python3
"""
Shared posture detection logic for PostureGuard.

This module contains the core metric extraction, baseline comparison,
and calibration I/O used by both the menu bar app and camera preview.
"""

import json
import os

import numpy as np

# --- Paths & camera ---
CALIBRATION_FILE = os.path.expanduser("~/posture_calibration.json")
CAMERA_INDEX = 0

# --- Detection thresholds (Medium/default) ---
THRESH_HEAD_DROP = 0.04
THRESH_HEAD_FORWARD = 0.03
THRESH_HEAD_TILT = 0.03
THRESH_SHOULDER_TILT = 0.025
THRESH_SLOUCH = 0.06

# --- Sensitivity presets ---
SENSITIVITY_PRESETS = {
    'low': {
        'THRESH_HEAD_DROP': 0.06,
        'THRESH_SLOUCH': 0.09,
        'THRESH_HEAD_FORWARD': 0.045,
        'THRESH_SHOULDER_TILT': 0.04,
    },
    'medium': {
        'THRESH_HEAD_DROP': 0.04,
        'THRESH_SLOUCH': 0.06,
        'THRESH_HEAD_FORWARD': 0.03,
        'THRESH_SHOULDER_TILT': 0.025,
    },
    'high': {
        'THRESH_HEAD_DROP': 0.025,
        'THRESH_SLOUCH': 0.04,
        'THRESH_HEAD_FORWARD': 0.02,
        'THRESH_SHOULDER_TILT': 0.015,
    },
}

# --- Penalty weights per issue ---
WEIGHT_HEAD_DROP = 30
WEIGHT_SLOUCH = 35
WEIGHT_LEAN = 20
WEIGHT_SHOULDER = 20
WEIGHT_FORWARD = 25

# --- Timing ---
CHECK_INTERVAL = 0.5  # seconds between posture checks
BAD_POSTURE_SECONDS = 5
COOLDOWN_SECONDS = 45

# --- Visibility ---
MIN_VISIBILITY = 0.4

# --- MediaPipe (lazy import — not needed for pure logic functions) ---
_mp = None
mp_pose = None
mp_face = None


def _init_mediapipe():
    """Lazy-load mediapipe on first use. Allows pure-logic tests without it."""
    global _mp, mp_pose, mp_face
    if _mp is None:
        import mediapipe as mp
        _mp = mp
        mp_pose = mp.solutions.pose
        mp_face = mp.solutions.face_mesh
    return mp_pose, mp_face


def extract_metrics(pose_landmarks, face_landmarks=None):
    """Extract normalized posture metrics from MediaPipe landmarks.

    Args:
        pose_landmarks: List of pose landmarks (from pose_landmarks.landmark).
        face_landmarks: Optional list of face mesh landmarks.

    Returns:
        Dict of metric name -> float, or None if key landmarks aren't visible.
    """
    pose, _ = _init_mediapipe()
    lm = pose_landmarks
    nose = lm[pose.PoseLandmark.NOSE]
    left_ear = lm[pose.PoseLandmark.LEFT_EAR]
    right_ear = lm[pose.PoseLandmark.RIGHT_EAR]
    left_shoulder = lm[pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = lm[pose.PoseLandmark.RIGHT_SHOULDER]

    if any(p.visibility < MIN_VISIBILITY for p in [nose, left_ear, right_ear, left_shoulder, right_shoulder]):
        return None

    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    mid_ear_y = (left_ear.y + right_ear.y) / 2

    face_tilt = 0.0
    face_forward_ratio = 0.0
    if face_landmarks:
        forehead = face_landmarks[10]
        chin = face_landmarks[152]
        face_tilt = forehead.x - chin.x
        left_cheek = face_landmarks[234]
        right_cheek = face_landmarks[454]
        face_forward_ratio = abs(left_cheek.x - right_cheek.x)

    return {
        "nose_to_shoulder_y": nose.y - mid_shoulder_y,
        "nose_to_shoulder_x": nose.x - mid_shoulder_x,
        "ear_shoulder_dist": mid_shoulder_y - mid_ear_y,
        "shoulder_tilt": left_shoulder.y - right_shoulder.y,
        "nose_to_ear_y": nose.y - mid_ear_y,
        "face_tilt": face_tilt,
        "face_forward_ratio": face_forward_ratio,
        "nose_y": nose.y,
        "mid_ear_y": mid_ear_y,
        "mid_shoulder_y": mid_shoulder_y,
    }


def compare_to_baseline(current, baseline, thresholds=None):
    """Compare current metrics against calibrated baseline.

    Args:
        current: Dict of current metric values.
        baseline: Dict of baseline metric values from calibration.
        thresholds: Optional dict overriding default thresholds.
            Keys: THRESH_HEAD_DROP, THRESH_SLOUCH, THRESH_HEAD_FORWARD, THRESH_SHOULDER_TILT.

    Returns:
        Tuple of (issues: list[str], score: int 0-100).
    """
    t_head = (thresholds or {}).get('THRESH_HEAD_DROP', THRESH_HEAD_DROP)
    t_slouch = (thresholds or {}).get('THRESH_SLOUCH', THRESH_SLOUCH)
    t_lean = (thresholds or {}).get('THRESH_HEAD_FORWARD', THRESH_HEAD_FORWARD)
    t_shoulder = (thresholds or {}).get('THRESH_SHOULDER_TILT', THRESH_SHOULDER_TILT)

    issues = []
    penalty = 0

    # Head drop
    head_drop = current["nose_to_shoulder_y"] - baseline["nose_to_shoulder_y"]
    if head_drop > t_head:
        severity = min(1.0, head_drop / (t_head * 3))
        penalty += severity * WEIGHT_HEAD_DROP
        issues.append(("Head dropping — chin up!", head_drop))

    # Slouch
    slouch = baseline["ear_shoulder_dist"] - current["ear_shoulder_dist"]
    if slouch > t_slouch:
        severity = min(1.0, slouch / (t_slouch * 3))
        penalty += severity * WEIGHT_SLOUCH
        issues.append(("Slouching — sit up straight!", slouch))

    # Lateral lean
    baseline_offset = baseline["nose_to_shoulder_x"]
    current_offset = current["nose_to_shoulder_x"]
    lateral_drift = abs(current_offset - baseline_offset)
    if lateral_drift > t_lean:
        severity = min(1.0, lateral_drift / (t_lean * 3))
        penalty += severity * WEIGHT_LEAN
        direction = "left" if (current_offset - baseline_offset) < 0 else "right"
        issues.append((f"Leaning {direction} — center up!", lateral_drift))

    # Shoulder tilt
    if abs(current["shoulder_tilt"]) > t_shoulder:
        tilt_diff = abs(current["shoulder_tilt"]) - abs(baseline["shoulder_tilt"])
        if tilt_diff > 0.01:
            severity = min(1.0, abs(current["shoulder_tilt"]) / (t_shoulder * 3))
            penalty += severity * WEIGHT_SHOULDER
            issues.append(("Shoulders uneven — level out!", abs(current["shoulder_tilt"])))

    # Forward lean (face width ratio)
    if current["face_forward_ratio"] > 0 and baseline.get("face_forward_ratio", 0) > 0:
        forward = current["face_forward_ratio"] - baseline["face_forward_ratio"]
        if forward > 0.03:
            severity = min(1.0, forward / 0.09)
            penalty += severity * WEIGHT_FORWARD
            issues.append(("Leaning forward — sit back!", forward))

    score = max(0, int(100 - penalty))
    return issues, score


def load_calibration():
    """Load calibration baseline from disk. Returns dict or None."""
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    return None


def save_calibration(metrics):
    """Save calibration baseline to disk."""
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)


def average_metrics(frames):
    """Average a list of metric dicts into a single baseline dict."""
    baseline = {}
    for key in frames[0]:
        baseline[key] = float(np.mean([f[key] for f in frames]))
    return baseline


def smooth_score(score_history, new_score, max_len=20):
    """Append a score and return the smoothed (averaged) value.

    Mutates score_history in place. Returns the smoothed int score.
    """
    score_history.append(new_score)
    if len(score_history) > max_len:
        score_history.pop(0)
    return int(np.mean(score_history))
