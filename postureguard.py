#!/usr/bin/env python3
"""
PostureGuard — invisible menu bar app that monitors your posture.

Sits in the menu bar. Watches via camera. Yells when you slouch.
No window, no dock icon, just silent judgment.
"""

import rumps
import cv2
import numpy as np
import subprocess
import threading
import time
import csv
import os
from datetime import datetime

from posture_core import (
    CALIBRATION_FILE, CAMERA_INDEX, CHECK_INTERVAL,
    BAD_POSTURE_SECONDS, COOLDOWN_SECONDS,
    SENSITIVITY_PRESETS, _init_mediapipe,
    extract_metrics, compare_to_baseline, load_calibration,
    save_calibration, average_metrics, smooth_score,
)

__version__ = "1.1.0"

LOG_DIR = os.path.expanduser("~/PostureGuard")
LOG_FILE = os.path.join(LOG_DIR, "posture_log.csv")


def say(text):
    subprocess.Popen(["say", "-v", "Samantha", "-r", "210", text])


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def log_posture(score, issues):
    """Append a timestamped posture reading to the CSV log."""
    _ensure_log_dir()
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "score", "issues"])
        issue_text = "; ".join(msg for msg, _ in issues) if issues else ""
        writer.writerow([datetime.now().isoformat(), score, issue_text])


def get_session_summary():
    """Return a summary string of today's posture stats."""
    if not os.path.exists(LOG_FILE):
        return "No session data yet"

    today = datetime.now().strftime("%Y-%m-%d")
    scores = []
    issue_counts = {}

    with open(LOG_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["timestamp"].startswith(today):
                continue
            try:
                scores.append(int(row["score"]))
            except (ValueError, KeyError):
                continue
            if row.get("issues"):
                for issue in row["issues"].split("; "):
                    label = issue.split("—")[0].strip() if "—" in issue else issue
                    issue_counts[label] = issue_counts.get(label, 0) + 1

    if not scores:
        return "No data today"

    avg = int(np.mean(scores))
    good_pct = int(sum(1 for s in scores if s > 80) / len(scores) * 100)
    top_issue = max(issue_counts, key=issue_counts.get) if issue_counts else "None"
    return f"Avg: {avg}% | Good: {good_pct}% of time | Top issue: {top_issue}"


class PostureGuardApp(rumps.App):
    def __init__(self):
        super().__init__(
            "PostureGuard",
            icon=None,
            title="PG",
            quit_button=None,
        )

        self.monitoring = False
        self.baseline = load_calibration()
        self.score = 100
        self.bad_posture_start = None
        self.last_yell_time = 0
        self.score_history = []
        self.last_issues = []
        self.monitor_thread = None
        self.stop_event = threading.Event()
        self.camera_proc = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.current_sensitivity = 'medium'
        self._log_counter = 0
        self._pending_ui = {}  # thread-safe UI updates

        # Menu items
        self.status_item = rumps.MenuItem("Status: Idle", callback=None)
        self.score_item = rumps.MenuItem("Score: --", callback=None)
        self.issues_item = rumps.MenuItem("No issues", callback=None)

        self.toggle_item = rumps.MenuItem("Start Monitoring", callback=self.toggle_monitoring)
        self.camera_item = rumps.MenuItem("Show Camera", callback=self.toggle_camera)
        self.calibrate_item = rumps.MenuItem("Calibrate Good Posture", callback=self.calibrate)

        self.sensitivity_menu = rumps.MenuItem("Sensitivity")
        self.sens_low = rumps.MenuItem("Low (relaxed)", callback=self.set_sensitivity_low)
        self.sens_med = rumps.MenuItem("Medium", callback=self.set_sensitivity_med)
        self.sens_high = rumps.MenuItem("High (strict)", callback=self.set_sensitivity_high)
        self.sensitivity_menu.update([self.sens_low, self.sens_med, self.sens_high])
        self.sens_med.state = True

        self.summary_item = rumps.MenuItem("Today's Summary", callback=self.show_summary)
        self.quit_item = rumps.MenuItem("Quit PostureGuard", callback=self.quit_app)

        self.menu = [
            self.status_item,
            self.score_item,
            self.issues_item,
            None,  # separator
            self.toggle_item,
            self.camera_item,
            self.calibrate_item,
            self.sensitivity_menu,
            None,  # separator
            self.summary_item,
            None,  # separator
            self.quit_item,
        ]

        # Auto-start if calibrated
        if self.baseline:
            self.start_monitoring()

    @rumps.timer(0.5)
    def _flush_ui(self, _):
        """Apply pending UI updates on the main thread."""
        pending = self._pending_ui
        if not pending:
            return
        self._pending_ui = {}
        if 'title' in pending:
            self.title = pending['title']
        if 'score' in pending:
            self.score_item.title = pending['score']
        if 'issues' in pending:
            self.issues_item.title = pending['issues']
        if 'camera_label' in pending:
            self.camera_item.title = pending['camera_label']

    def _get_thresholds(self):
        return SENSITIVITY_PRESETS[self.current_sensitivity]

    def _set_sensitivity(self, level, label):
        self.current_sensitivity = level
        self.sens_low.state = (level == 'low')
        self.sens_med.state = (level == 'medium')
        self.sens_high.state = (level == 'high')
        rumps.notification("PostureGuard", "", f"Sensitivity: {label}")

    def set_sensitivity_low(self, _):
        self._set_sensitivity('low', "Low (relaxed)")

    def set_sensitivity_med(self, _):
        self._set_sensitivity('medium', "Medium")

    def set_sensitivity_high(self, _):
        self._set_sensitivity('high', "High (strict)")

    def show_summary(self, _):
        summary = get_session_summary()
        rumps.notification("PostureGuard — Today", "", summary)

    def toggle_camera(self, _):
        if self.camera_proc and self.camera_proc.poll() is None:
            self.camera_proc.terminate()
            self.camera_proc = None
            self.camera_item.title = "Show Camera"
        else:
            preview_script = os.path.join(self.script_dir, "camera_preview.py")
            if not os.path.exists(preview_script):
                rumps.notification("PostureGuard", "Error", f"Camera preview not found: {preview_script}")
                return
            self.camera_proc = subprocess.Popen(
                ["python3", preview_script],
                stderr=subprocess.PIPE,
            )
            # Check if it died immediately
            threading.Thread(target=self._watch_camera_proc, daemon=True).start()
            self.camera_item.title = "Hide Camera"

    def _watch_camera_proc(self):
        """Watch camera subprocess for early failure."""
        proc = self.camera_proc
        if proc is None:
            return
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode(errors='replace').strip() if proc.stderr else ""
            msg = stderr[-200:] if stderr else f"Exit code {proc.returncode}"
            rumps.notification("PostureGuard", "Camera preview failed", msg)
        self.camera_proc = None
        self._pending_ui['camera_label'] = "Show Camera"

    def toggle_monitoring(self, _):
        if self.monitoring:
            self.stop_monitoring()
        else:
            if not self.baseline:
                rumps.notification("PostureGuard", "", "Please calibrate first! Sit straight and click 'Calibrate Good Posture'.")
                return
            self.start_monitoring()

    def start_monitoring(self):
        self.monitoring = True
        self.stop_event.clear()
        self.toggle_item.title = "Stop Monitoring"
        self.title = "PG"
        self.status_item.title = "Status: Monitoring"
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        self.stop_event.set()
        self.toggle_item.title = "Start Monitoring"
        self.title = "PG"
        self.status_item.title = "Status: Paused"
        self.score_item.title = "Score: --"
        self.issues_item.title = "No issues"

    def calibrate(self, _):
        """Run calibration in background thread."""
        if self.monitoring:
            self.stop_monitoring()
            time.sleep(1)

        rumps.notification("PostureGuard", "Calibrating...", "Sit up straight! Capturing your good posture for 3 seconds...")
        say("Sit up straight. Calibrating in 3 seconds.")

        def _do_calibrate():
            time.sleep(2)  # give user time to sit up
            _mp_pose, _mp_face = _init_mediapipe()

            pose_detector = _mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            face_detector = _mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )

            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                rumps.notification("PostureGuard", "Error", "Cannot access camera!")
                return

            try:
                frames = []
                for _ in range(45):  # ~3 seconds at ~15fps effective
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    pose_r = pose_detector.process(rgb)
                    face_r = face_detector.process(rgb)

                    if pose_r.pose_landmarks:
                        face_lm = None
                        if face_r.multi_face_landmarks:
                            face_lm = face_r.multi_face_landmarks[0].landmark
                        m = extract_metrics(pose_r.pose_landmarks.landmark, face_lm)
                        if m:
                            frames.append(m)
                    time.sleep(0.066)
            finally:
                cap.release()
                pose_detector.close()
                face_detector.close()

            if len(frames) < 10:
                rumps.notification("PostureGuard", "Error", "Couldn't detect your pose. Make sure camera can see your face and shoulders.")
                say("Calibration failed. I can't see you properly.")
                return

            self.baseline = average_metrics(frames)
            save_calibration(self.baseline)

            say("Calibration complete. I'm watching you.")
            rumps.notification("PostureGuard", "Calibrated!", f"Captured {len(frames)} frames. Now monitoring.")
            self.score_history = []
            self.start_monitoring()

        threading.Thread(target=_do_calibrate, daemon=True).start()

    def _monitor_loop(self):
        """Background loop: grab frame, check posture, update menu bar."""
        _mp_pose, _mp_face = _init_mediapipe()
        pose_detector = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        face_detector = _mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            rumps.notification("PostureGuard", "Error", "Cannot access camera!")
            self.monitoring = False
            return

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                pose_r = pose_detector.process(rgb)
                face_r = face_detector.process(rgb)

                if pose_r.pose_landmarks:
                    face_lm = None
                    if face_r.multi_face_landmarks:
                        face_lm = face_r.multi_face_landmarks[0].landmark

                    metrics = extract_metrics(pose_r.pose_landmarks.landmark, face_lm)

                    if metrics and self.baseline:
                        thresholds = self._get_thresholds()
                        issues, score = compare_to_baseline(metrics, self.baseline, thresholds)
                        smoothed = smooth_score(self.score_history, score)

                        self.score = smoothed
                        self.last_issues = issues

                        # Log every 6th reading (~3 seconds) to avoid huge files
                        self._log_counter += 1
                        if self._log_counter % 6 == 0:
                            log_posture(smoothed, issues)

                        # Queue UI updates for the main thread
                        if smoothed > 80:
                            ui_title = "PG"
                        elif smoothed > 50:
                            ui_title = "PG!"
                        else:
                            ui_title = "PG!!"

                        ui_issues = issues[0][0].split("—")[0].strip() if issues else "Looking good!"
                        self._pending_ui = {
                            'title': ui_title,
                            'score': f"Score: {smoothed}%",
                            'issues': ui_issues,
                        }

                        # Yell logic
                        now = time.time()
                        if issues:
                            if self.bad_posture_start is None:
                                self.bad_posture_start = now
                            elif (now - self.bad_posture_start > BAD_POSTURE_SECONDS
                                  and now - self.last_yell_time > COOLDOWN_SECONDS):
                                msg = issues[0][0]
                                yell = msg.split("—")[1].strip().rstrip("!") + "!" if "—" in msg else "Fix your posture!"
                                say(f"Hey! {yell}")
                                rumps.notification("PostureGuard", f"Score: {smoothed}%", msg)
                                self.last_yell_time = now
                        else:
                            self.bad_posture_start = None

                time.sleep(CHECK_INTERVAL)
        finally:
            cap.release()
            pose_detector.close()
            face_detector.close()

    def quit_app(self, _):
        self.stop_monitoring()
        if self.camera_proc and self.camera_proc.poll() is None:
            self.camera_proc.terminate()
        rumps.quit_application()


if __name__ == "__main__":
    PostureGuardApp().run()
