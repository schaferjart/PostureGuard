#!/usr/bin/env python3
"""Standalone camera preview window for PostureGuard. Launched as subprocess."""

import cv2
import mediapipe as mp
import time

from posture_core import (
    CAMERA_INDEX, _init_mediapipe,
    extract_metrics, compare_to_baseline, load_calibration, smooth_score,
)

mp_draw = mp.solutions.drawing_utils


def main():
    _mp_pose, _mp_face = _init_mediapipe()
    baseline = load_calibration()

    pose = _mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6,
    )
    face = _mp_face.FaceMesh(
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

    try:
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
                    frame, pose_r.pose_landmarks, _mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2),
                )
                if face_r.multi_face_landmarks:
                    for fl in face_r.multi_face_landmarks:
                        mp_draw.draw_landmarks(
                            frame, fl, _mp_face.FACEMESH_CONTOURS,
                            mp_draw.DrawingSpec(color=(0, 180, 180), thickness=1, circle_radius=1),
                            mp_draw.DrawingSpec(color=(0, 150, 150), thickness=1),
                        )

                face_lm = face_r.multi_face_landmarks[0].landmark if face_r.multi_face_landmarks else None
                metrics = extract_metrics(pose_r.pose_landmarks.landmark, face_lm)

                if metrics and baseline:
                    issues, score = compare_to_baseline(metrics, baseline)
                    score = smooth_score(score_history, score, max_len=30)

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
                for i, (label, val) in enumerate(issues):
                    display = f"{label.split('—')[0].strip().upper()} ({val:.3f})"
                    cv2.putText(frame, display, (10, 100 + i * 28),
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
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        face.close()


if __name__ == "__main__":
    main()
