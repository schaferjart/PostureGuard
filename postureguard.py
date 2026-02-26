#!/usr/bin/env python3
"""
PostureGuard — invisible menu bar app that monitors your posture.

Sits in the menu bar. Watches via camera. Yells when you slouch.
No window, no dock icon, just silent judgment.
"""

import rumps
import cv2
import mediapipe as mp
import numpy as np
import subprocess
import threading
import time
import json
import os

CALIBRATION_FILE = os.path.expanduser("~/posture_calibration.json")
CAMERA_INDEX = 0
CHECK_INTERVAL = 0.5  # seconds between posture checks (saves CPU vs every frame)

# Thresholds
THRESH_HEAD_DROP = 0.04
THRESH_HEAD_FORWARD = 0.03
THRESH_HEAD_TILT = 0.03
THRESH_SHOULDER_TILT = 0.025
THRESH_SLOUCH = 0.06

BAD_POSTURE_SECONDS = 5
COOLDOWN_SECONDS = 45

# MediaPipe setup (lazy init in background thread)
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh


def say(text):
    subprocess.Popen(["say", "-v", "Samantha", "-r", "210", text])


def extract_metrics(pose_landmarks, face_landmarks=None):
    lm = pose_landmarks
    nose = lm[mp_pose.PoseLandmark.NOSE]
    left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]
    left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    if any(p.visibility < 0.4 for p in [nose, left_ear, right_ear, left_shoulder, right_shoulder]):
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


def compare_to_baseline(current, baseline):
    issues = []
    penalty = 0

    head_drop = current["nose_to_shoulder_y"] - baseline["nose_to_shoulder_y"]
    if head_drop > THRESH_HEAD_DROP:
        severity = min(1.0, head_drop / (THRESH_HEAD_DROP * 3))
        penalty += severity * 30
        issues.append("Head dropping — chin up!")

    slouch = baseline["ear_shoulder_dist"] - current["ear_shoulder_dist"]
    if slouch > THRESH_SLOUCH:
        severity = min(1.0, slouch / (THRESH_SLOUCH * 3))
        penalty += severity * 35
        issues.append("Slouching — sit up straight!")

    baseline_offset = baseline["nose_to_shoulder_x"]
    current_offset = current["nose_to_shoulder_x"]
    lateral_drift = abs(current_offset - baseline_offset)
    if lateral_drift > THRESH_HEAD_FORWARD:
        severity = min(1.0, lateral_drift / (THRESH_HEAD_FORWARD * 3))
        penalty += severity * 20
        direction = "left" if (current_offset - baseline_offset) < 0 else "right"
        issues.append(f"Leaning {direction} — center up!")

    if abs(current["shoulder_tilt"]) > THRESH_SHOULDER_TILT:
        tilt_diff = abs(current["shoulder_tilt"]) - abs(baseline["shoulder_tilt"])
        if tilt_diff > 0.01:
            severity = min(1.0, abs(current["shoulder_tilt"]) / (THRESH_SHOULDER_TILT * 3))
            penalty += severity * 20
            issues.append("Shoulders uneven — level out!")

    if current["face_forward_ratio"] > 0 and baseline.get("face_forward_ratio", 0) > 0:
        forward = current["face_forward_ratio"] - baseline["face_forward_ratio"]
        if forward > 0.03:
            severity = min(1.0, forward / 0.09)
            penalty += severity * 25
            issues.append("Leaning forward — sit back!")

    score = max(0, int(100 - penalty))
    return issues, score


def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    return None


def save_calibration(metrics):
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)


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

        # Menu items
        self.status_item = rumps.MenuItem("Status: Idle", callback=None)
        self.score_item = rumps.MenuItem("Score: --", callback=None)
        self.issues_item = rumps.MenuItem("No issues", callback=None)
        self.separator1 = None

        if self.baseline:
            self.toggle_item = rumps.MenuItem("Start Monitoring", callback=self.toggle_monitoring)
        else:
            self.toggle_item = rumps.MenuItem("Start Monitoring", callback=self.toggle_monitoring)

        self.camera_item = rumps.MenuItem("Show Camera", callback=self.toggle_camera)
        self.calibrate_item = rumps.MenuItem("Calibrate Good Posture", callback=self.calibrate)
        self.sensitivity_menu = rumps.MenuItem("Sensitivity")
        self.sens_low = rumps.MenuItem("Low (relaxed)", callback=self.set_sensitivity_low)
        self.sens_med = rumps.MenuItem("Medium", callback=self.set_sensitivity_med)
        self.sens_high = rumps.MenuItem("High (strict)", callback=self.set_sensitivity_high)
        self.sensitivity_menu.update([self.sens_low, self.sens_med, self.sens_high])
        self.sens_med.state = True

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
            self.quit_item,
        ]

        # Auto-start if calibrated
        if self.baseline:
            self.start_monitoring()

    def set_sensitivity_low(self, _):
        global THRESH_HEAD_DROP, THRESH_SLOUCH, THRESH_HEAD_FORWARD, THRESH_SHOULDER_TILT
        THRESH_HEAD_DROP = 0.06
        THRESH_SLOUCH = 0.09
        THRESH_HEAD_FORWARD = 0.045
        THRESH_SHOULDER_TILT = 0.04
        self.sens_low.state = True
        self.sens_med.state = False
        self.sens_high.state = False
        rumps.notification("PostureGuard", "", "Sensitivity: Low (relaxed)")

    def set_sensitivity_med(self, _):
        global THRESH_HEAD_DROP, THRESH_SLOUCH, THRESH_HEAD_FORWARD, THRESH_SHOULDER_TILT
        THRESH_HEAD_DROP = 0.04
        THRESH_SLOUCH = 0.06
        THRESH_HEAD_FORWARD = 0.03
        THRESH_SHOULDER_TILT = 0.025
        self.sens_low.state = False
        self.sens_med.state = True
        self.sens_high.state = False
        rumps.notification("PostureGuard", "", "Sensitivity: Medium")

    def set_sensitivity_high(self, _):
        global THRESH_HEAD_DROP, THRESH_SLOUCH, THRESH_HEAD_FORWARD, THRESH_SHOULDER_TILT
        THRESH_HEAD_DROP = 0.025
        THRESH_SLOUCH = 0.04
        THRESH_HEAD_FORWARD = 0.02
        THRESH_SHOULDER_TILT = 0.015
        self.sens_low.state = False
        self.sens_med.state = False
        self.sens_high.state = True
        rumps.notification("PostureGuard", "", "Sensitivity: High (strict)")

    def toggle_camera(self, _):
        if self.camera_proc and self.camera_proc.poll() is None:
            # Camera is running, kill it
            self.camera_proc.terminate()
            self.camera_proc = None
            self.camera_item.title = "Show Camera"
        else:
            # Launch camera preview as separate process
            preview_script = os.path.join(self.script_dir, "camera_preview.py")
            self.camera_proc = subprocess.Popen(["python3", preview_script])
            self.camera_item.title = "Hide Camera"

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

            pose_detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            face_detector = mp_face.FaceMesh(
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

            cap.release()
            pose_detector.close()
            face_detector.close()

            if len(frames) < 10:
                rumps.notification("PostureGuard", "Error", "Couldn't detect your pose. Make sure camera can see your face and shoulders.")
                say("Calibration failed. I can't see you properly.")
                return

            self.baseline = {}
            for key in frames[0]:
                self.baseline[key] = float(np.mean([f[key] for f in frames]))
            save_calibration(self.baseline)

            say("Calibration complete. I'm watching you.")
            rumps.notification("PostureGuard", "Calibrated!", f"Captured {len(frames)} frames. Now monitoring.")
            self.score_history = []
            self.start_monitoring()

        threading.Thread(target=_do_calibrate, daemon=True).start()

    def _monitor_loop(self):
        """Background loop: grab frame, check posture, update menu bar."""
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        face_detector = mp_face.FaceMesh(
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
                    issues, score = compare_to_baseline(metrics, self.baseline)

                    self.score_history.append(score)
                    if len(self.score_history) > 20:
                        self.score_history.pop(0)
                    smoothed = int(np.mean(self.score_history))

                    self.score = smoothed
                    self.last_issues = issues

                    # Update menu bar icon
                    if smoothed > 80:
                        self.title = "PG"
                    elif smoothed > 50:
                        self.title = "PG!"
                    else:
                        self.title = "PG!!"

                    # Update menu items
                    self.score_item.title = f"Score: {smoothed}%"
                    if issues:
                        self.issues_item.title = issues[0].split("—")[0].strip()
                    else:
                        self.issues_item.title = "Looking good!"

                    # Yell logic
                    now = time.time()
                    if issues:
                        if self.bad_posture_start is None:
                            self.bad_posture_start = now
                        elif (now - self.bad_posture_start > BAD_POSTURE_SECONDS
                              and now - self.last_yell_time > COOLDOWN_SECONDS):
                            msg = issues[0]
                            yell = msg.split("—")[1].strip().rstrip("!") + "!" if "—" in msg else "Fix your posture!"
                            say(f"Hey! {yell}")
                            rumps.notification("PostureGuard", f"Score: {smoothed}%", msg)
                            self.last_yell_time = now
                    else:
                        self.bad_posture_start = None

            time.sleep(CHECK_INTERVAL)

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
