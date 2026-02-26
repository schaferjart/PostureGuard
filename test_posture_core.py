#!/usr/bin/env python3
"""Tests for posture_core — the shared detection logic."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from posture_core import (
    SENSITIVITY_PRESETS,
    THRESH_HEAD_DROP, THRESH_SLOUCH, THRESH_HEAD_FORWARD, THRESH_SHOULDER_TILT,
    WEIGHT_HEAD_DROP, WEIGHT_SLOUCH, WEIGHT_LEAN, WEIGHT_SHOULDER, WEIGHT_FORWARD,
    compare_to_baseline,
    load_calibration,
    save_calibration,
    average_metrics,
    smooth_score,
)


# -- Helpers --

def make_baseline():
    """A representative calibrated baseline."""
    return {
        "nose_to_shoulder_y": -0.25,
        "nose_to_shoulder_x": 0.0,
        "ear_shoulder_dist": 0.18,
        "shoulder_tilt": 0.0,
        "nose_to_ear_y": -0.05,
        "face_tilt": 0.0,
        "face_forward_ratio": 0.12,
        "nose_y": 0.35,
        "mid_ear_y": 0.40,
        "mid_shoulder_y": 0.60,
    }


def make_current(overrides=None):
    """Current metrics identical to baseline (perfect posture), with optional overrides."""
    m = make_baseline()
    if overrides:
        m.update(overrides)
    return m


class TestCompareToBaseline(unittest.TestCase):
    """Unit tests for compare_to_baseline()."""

    def test_perfect_posture_returns_no_issues(self):
        baseline = make_baseline()
        current = make_current()
        issues, score = compare_to_baseline(current, baseline)
        self.assertEqual(issues, [])
        self.assertEqual(score, 100)

    def test_head_drop_detected(self):
        baseline = make_baseline()
        # Simulate head dropping: nose_to_shoulder_y increases
        current = make_current({"nose_to_shoulder_y": baseline["nose_to_shoulder_y"] + 0.08})
        issues, score = compare_to_baseline(current, baseline)
        labels = [msg for msg, _ in issues]
        self.assertTrue(any("Head dropping" in l for l in labels))
        self.assertLess(score, 100)

    def test_slouch_detected(self):
        baseline = make_baseline()
        # Simulate slouching: ear_shoulder_dist decreases
        current = make_current({"ear_shoulder_dist": baseline["ear_shoulder_dist"] - 0.10})
        issues, score = compare_to_baseline(current, baseline)
        labels = [msg for msg, _ in issues]
        self.assertTrue(any("Slouching" in l for l in labels))
        self.assertLess(score, 100)

    def test_lateral_lean_detected(self):
        baseline = make_baseline()
        # Simulate leaning right: nose_to_shoulder_x increases
        current = make_current({"nose_to_shoulder_x": baseline["nose_to_shoulder_x"] + 0.06})
        issues, score = compare_to_baseline(current, baseline)
        labels = [msg for msg, _ in issues]
        self.assertTrue(any("Leaning right" in l for l in labels))

    def test_lateral_lean_left_detected(self):
        baseline = make_baseline()
        current = make_current({"nose_to_shoulder_x": baseline["nose_to_shoulder_x"] - 0.06})
        issues, score = compare_to_baseline(current, baseline)
        labels = [msg for msg, _ in issues]
        self.assertTrue(any("Leaning left" in l for l in labels))

    def test_shoulder_tilt_detected(self):
        baseline = make_baseline()
        # Simulate uneven shoulders
        current = make_current({"shoulder_tilt": 0.06})
        issues, score = compare_to_baseline(current, baseline)
        labels = [msg for msg, _ in issues]
        self.assertTrue(any("Shoulders uneven" in l for l in labels))

    def test_forward_lean_detected(self):
        baseline = make_baseline()
        # Simulate leaning forward: face_forward_ratio increases significantly
        current = make_current({"face_forward_ratio": baseline["face_forward_ratio"] + 0.05})
        issues, score = compare_to_baseline(current, baseline)
        labels = [msg for msg, _ in issues]
        self.assertTrue(any("Leaning forward" in l for l in labels))

    def test_forward_lean_ignored_without_baseline_face(self):
        baseline = make_baseline()
        baseline["face_forward_ratio"] = 0  # no face data in calibration
        current = make_current({"face_forward_ratio": 0.20})
        issues, score = compare_to_baseline(current, baseline)
        labels = [msg for msg, _ in issues]
        self.assertFalse(any("forward" in l.lower() for l in labels))

    def test_score_never_below_zero(self):
        baseline = make_baseline()
        # Everything terrible at once
        current = make_current({
            "nose_to_shoulder_y": baseline["nose_to_shoulder_y"] + 0.30,
            "ear_shoulder_dist": baseline["ear_shoulder_dist"] - 0.30,
            "nose_to_shoulder_x": 0.30,
            "shoulder_tilt": 0.20,
            "face_forward_ratio": baseline["face_forward_ratio"] + 0.20,
        })
        issues, score = compare_to_baseline(current, baseline)
        self.assertGreaterEqual(score, 0)

    def test_custom_thresholds_override_defaults(self):
        baseline = make_baseline()
        # This deviation is below default threshold but above a very strict one
        small_drop = THRESH_HEAD_DROP * 0.5
        current = make_current({"nose_to_shoulder_y": baseline["nose_to_shoulder_y"] + small_drop})

        # Default: no issue
        issues, _ = compare_to_baseline(current, baseline)
        self.assertEqual(issues, [])

        # Strict threshold: issue detected
        strict = {'THRESH_HEAD_DROP': small_drop * 0.5}
        issues, _ = compare_to_baseline(current, baseline, thresholds=strict)
        labels = [msg for msg, _ in issues]
        self.assertTrue(any("Head dropping" in l for l in labels))

    def test_sensitivity_presets_are_complete(self):
        required_keys = {'THRESH_HEAD_DROP', 'THRESH_SLOUCH', 'THRESH_HEAD_FORWARD', 'THRESH_SHOULDER_TILT'}
        for name, preset in SENSITIVITY_PRESETS.items():
            self.assertEqual(set(preset.keys()), required_keys, f"Preset '{name}' has wrong keys")

    def test_low_sensitivity_is_more_permissive(self):
        baseline = make_baseline()
        drop = 0.05  # triggers on medium but not on low
        current = make_current({"nose_to_shoulder_y": baseline["nose_to_shoulder_y"] + drop})

        issues_med, score_med = compare_to_baseline(current, baseline, SENSITIVITY_PRESETS['medium'])
        issues_low, score_low = compare_to_baseline(current, baseline, SENSITIVITY_PRESETS['low'])
        self.assertGreaterEqual(score_low, score_med)

    def test_multiple_issues_compound_penalty(self):
        baseline = make_baseline()
        current = make_current({
            "nose_to_shoulder_y": baseline["nose_to_shoulder_y"] + 0.08,
            "ear_shoulder_dist": baseline["ear_shoulder_dist"] - 0.10,
        })
        issues, score = compare_to_baseline(current, baseline)
        self.assertGreater(len(issues), 1)
        # Score should be lower than single-issue case
        single = make_current({"nose_to_shoulder_y": baseline["nose_to_shoulder_y"] + 0.08})
        _, single_score = compare_to_baseline(single, baseline)
        self.assertLess(score, single_score)

    def test_issues_include_deviation_value(self):
        baseline = make_baseline()
        current = make_current({"nose_to_shoulder_y": baseline["nose_to_shoulder_y"] + 0.08})
        issues, _ = compare_to_baseline(current, baseline)
        self.assertGreater(len(issues), 0)
        msg, val = issues[0]
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0)


class TestCalibrationIO(unittest.TestCase):
    """Tests for save/load calibration."""

    def test_roundtrip(self):
        baseline = make_baseline()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp = f.name
        try:
            with patch('posture_core.CALIBRATION_FILE', tmp):
                save_calibration(baseline)
                loaded = load_calibration()
            self.assertEqual(baseline, loaded)
        finally:
            os.unlink(tmp)

    def test_load_missing_file_returns_none(self):
        with patch('posture_core.CALIBRATION_FILE', '/tmp/nonexistent_postureguard_test.json'):
            self.assertIsNone(load_calibration())


class TestAverageMetrics(unittest.TestCase):
    """Tests for average_metrics()."""

    def test_single_frame(self):
        frames = [make_baseline()]
        result = average_metrics(frames)
        for key, val in make_baseline().items():
            self.assertAlmostEqual(result[key], val, places=6)

    def test_multiple_frames_averaged(self):
        f1 = make_current({"nose_y": 0.30})
        f2 = make_current({"nose_y": 0.40})
        result = average_metrics([f1, f2])
        self.assertAlmostEqual(result["nose_y"], 0.35, places=6)

    def test_output_values_are_floats(self):
        result = average_metrics([make_baseline()])
        for val in result.values():
            self.assertIsInstance(val, float)


class TestSmoothScore(unittest.TestCase):
    """Tests for smooth_score()."""

    def test_single_score(self):
        history = []
        result = smooth_score(history, 80)
        self.assertEqual(result, 80)
        self.assertEqual(history, [80])

    def test_smoothing_averages(self):
        history = [100, 100, 100, 100]
        result = smooth_score(history, 50)
        # Average of [100, 100, 100, 100, 50] = 90
        self.assertEqual(result, 90)

    def test_max_len_enforced(self):
        history = list(range(5))
        smooth_score(history, 99, max_len=5)
        # After adding 1 and removing 1, length stays at max_len
        self.assertEqual(len(history), 5)
        # Oldest element (0) should have been removed
        self.assertNotIn(0, history)
        self.assertIn(99, history)

    def test_returns_int(self):
        history = []
        result = smooth_score(history, 77)
        self.assertIsInstance(result, int)


if __name__ == "__main__":
    unittest.main()
