#!/usr/bin/env python3
"""Standalone camera preview window for PostureGuard. Launched as subprocess."""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

CALIBRATION_FILE = os.path.expanduser("~/posture_calibration.json")
CAMERA_INDEX = 0

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

THRESH_HEAD_DROP = 0.04
THRESH_SLOUCH = 0.06
THRESH_HEAD_FORWARD = 0.03
THRESH_SHOULDER_TILT = 0.025


def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    return None


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
        face_tilt = face_landmarks[10].x - face_landmarks[152].x
        face_forward_ratio = abs(face_landmarks[234].x - face_landmarks[454].x)

    return {
        "nose_to_shoulder_y": nose.y - mid_shoulder_y,
        "nose_to_shoulder_x": nose.x - mid_shoulder_x,
        "ear_shoulder_dist": mid_shoulder_y - mid_ear_y,
        "shoulder_tilt": left_shoulder.y - right_shoulder.y,
        "nose_to_ear_y": nose.y - mid_ear_y,
        "face_tilt": face_tilt,
        "face_forward_ratio": face_forward_ratio,
    }


def compare_to_baseline(current, baseline):
    issues = []
    penalty = 0

    head_drop = current["nose_to_shoulder_y"] - baseline["nose_to_shoulder_y"]
    if head_drop > THRESH_HEAD_DROP:
        penalty += min(1.0, head_drop / (THRESH_HEAD_DROP * 3)) * 30
        issues.append(f"HEAD DROPPING ({head_drop:.3f})")

    slouch = baseline["ear_shoulder_dist"] - current["ear_shoulder_dist"]
    if slouch > THRESH_SLOUCH:
        penalty += min(1.0, slouch / (THRESH_SLOUCH * 3)) * 35
        issues.append(f"SLOUCHING ({slouch:.3f})")

    lateral_drift = abs(current["nose_to_shoulder_x"] - baseline["nose_to_shoulder_x"])
    if lateral_drift > THRESH_HEAD_FORWARD:
        penalty += min(1.0, lateral_drift / (THRESH_HEAD_FORWARD * 3)) * 20
        direction = "LEFT" if (current["nose_to_shoulder_x"] - baseline["nose_to_shoulder_x"]) < 0 else "RIGHT"
        issues.append(f"LEANING {direction} ({lateral_drift:.3f})")

    if abs(current["shoulder_tilt"]) > THRESH_SHOULDER_TILT:
        tilt_diff = abs(current["shoulder_tilt"]) - abs(baseline["shoulder_tilt"])
        if tilt_diff > 0.01:
            penalty += min(1.0, abs(current["shoulder_tilt"]) / (THRESH_SHOULDER_TILT * 3)) * 20
            issues.append("SHOULDERS UNEVEN")

    if baseline.get("face_forward_ratio", 0) > 0 and current["face_forward_ratio"] > 0:
        forward = current["face_forward_ratio"] - baseline["face_forward_ratio"]
        if forward > 0.03:
            penalty += min(1.0, forward / 0.09) * 25
            issues.append(f"LEANING FORWARD ({forward:.3f})")

    return issues, max(0, int(100 - penalty))


def main():
    baseline = load_calibration()

    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6,
    )
    face = mp_face.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    score_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_r = pose.process(rgb)
        face_r = face.process(rgb)

        issues = []
        score = 100

        if pose_r.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, pose_r.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2),
            )
            if face_r.multi_face_landmarks:
                for fl in face_r.multi_face_landmarks:
                    mp_draw.draw_landmarks(
                        frame, fl, mp_face.FACEMESH_CONTOURS,
                        mp_draw.DrawingSpec(color=(0, 180, 180), thickness=1, circle_radius=1),
                        mp_draw.DrawingSpec(color=(0, 150, 150), thickness=1),
                    )

            face_lm = face_r.multi_face_landmarks[0].landmark if face_r.multi_face_landmarks else None
            metrics = extract_metrics(pose_r.pose_landmarks.landmark, face_lm)

            if metrics and baseline:
                issues, score = compare_to_baseline(metrics, baseline)
                score_history.append(score)
                if len(score_history) > 30:
                    score_history.pop(0)
                score = int(np.mean(score_history))

        # HUD
        bar_color = (0, 255, 0) if score > 70 else (0, 200, 255) if score > 40 else (0, 0, 255)
        cv2.rectangle(frame, (10, 10), (10 + int(score * 2.5), 40), bar_color, -1)
        cv2.rectangle(frame, (10, 10), (260, 40), (255, 255, 255), 2)
        cv2.putText(frame, f"POSTURE: {score}%", (15, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if issues:
            cv2.putText(frame, "BAD POSTURE", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if int(time.time() * 2) % 2:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            for i, issue in enumerate(issues):
                cv2.putText(frame, issue, (10, 100 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "LOOKING GOOD", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if not baseline:
            cv2.putText(frame, "NO CALIBRATION — use menu bar to calibrate", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("PostureGuard Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    face.close()


if __name__ == "__main__":
    main()
