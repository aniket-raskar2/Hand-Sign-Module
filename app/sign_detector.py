"""
sign_detector.py
================
Drop this file into your Django `app/` folder alongside views.py.

This module handles ALL the mediapipe + model inference logic so that
views.py stays clean and future label changes only require editing
LABELS_DISPLAY_MAP at the bottom of this file.

When you retrain with new/more words:
  1. Replace  app/models/model.h5
  2. Replace  app/data/labels.txt
  3. Update   LABELS_DISPLAY_MAP  (add your new label keys)
  That's it — nothing else needs changing.
"""

import os
import sys
import numpy as np
import cv2
import collections

# ── Mediapipe import ───────────────────────────────────────────────────────
try:
    import mediapipe as mp
    mp_hands_mod  = mp.solutions.hands
    mp_face_mod   = mp.solutions.face_mesh
    mp_draw       = mp.solutions.drawing_utils
    mp_draw_style = mp.solutions.drawing_styles
except AttributeError:
    raise ImportError(
        "mediapipe 'solutions' API missing. "
        "Run: pip install mediapipe==0.10.9"
    )

import tensorflow as tf

# ── Paths — relative to this file (app/) ──────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE, "model", "model.h5")
LABELS_PATH = os.path.join(BASE, "model", "labels.txt")
# ── Detection config ───────────────────────────────────────────────────────
CONF_THRESHOLD = 0.75   # minimum confidence to accept a prediction
VOTE_WINDOW    = 10     # rolling vote window (same as 3_interface.py)
MIN_DETECT     = 0.7
MIN_TRACK      = 0.6
IMG_W          = 1280   # MUST match 1_collect_data.py capture resolution
IMG_H          = 720

# ── Colors ─────────────────────────────────────────────────────────────────
GREEN  = (60,  220, 80)
ORANGE = (0,   165, 255)
RED    = (60,  60,  220)
WHITE  = (255, 255, 255)
DARK   = (25,  25,  25)
HAND_R = (50,  200, 255)   # right hand — cyan
HAND_L = (255, 180, 50)    # left hand  — amber
FACE_C = (180, 180, 180)   # face mesh  — grey

# ── DISPLAY MAP ────────────────────────────────────────────────────────────
# Key   = label string as written in labels.txt  (lowercase / as trained)
# Value = what to show on screen and add to caption
#
# THIS IS THE ONLY THING YOU NEED TO UPDATE when you retrain with new words.
# Add one line per new label.  Keys must match labels.txt exactly.
#
LABELS_DISPLAY_MAP = {
    "hello":     "Hello",
    "thanks":    "Thanks",
    "teacher":   "Teacher",
    "indian":    "Indian",
    "i_am":      "I am",
    "you_are":   "You are",
    "beautiful": "Beautiful",
    "good":      "Good",
    "practice":  "Practice",
    "man":       "Man",
    "woman":     "Woman",
    "place":     "Place",
    "time":      "Time",
    "marry":     "Marry",
    "house":     "House",
    "food":      "Food",
    "0":         "0",
    "1":         "1",
    "2":         "2",
    "3":         "3",
    "4":         "4",
    "5":         "5",
    "6":         "6",
    "7":         "7",
    "8":         "8",
    "9":         "9",
}

# ── Keypoint helpers (mirrors utils.py exactly) ────────────────────────────

FACE_KEY_IDS = [
    33,  133, 362, 263, 1,   4,
    61,  291, 0,   17,  152, 10,
    234, 454, 70,  300, 105, 334,
    6,   168, 195, 5,   98,  327,
]

def _normalize_hand(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    span = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts /= span
    return pts.flatten().tolist()

def _normalize_face(landmarks, w, h):
    pts = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in FACE_KEY_IDS],
        dtype=np.float32,
    )
    pts -= pts.mean(axis=0)
    span = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts /= span
    return pts.flatten().tolist()

def _build_feature_vector(hand_res, face_res, w, h):
    right_vec = [0.0] * 63
    left_vec  = [0.0] * 63
    if hand_res.multi_hand_landmarks:
        for lms, hd in zip(hand_res.multi_hand_landmarks,
                           hand_res.multi_handedness):
            vec = _normalize_hand(lms.landmark)
            if hd.classification[0].label == "Left":
                right_vec = vec
            else:
                left_vec = vec
    face_vec = (
        _normalize_face(face_res.multi_face_landmarks[0].landmark, w, h)
        if face_res.multi_face_landmarks
        else [0.0] * 48
    )
    return right_vec + left_vec + face_vec


# ── Landmark drawing (same colours as 3_interface.py) ─────────────────────

def _draw_hand_landmarks(frame, hand_res):
    if not hand_res.multi_hand_landmarks:
        return
    for lms, hd in zip(hand_res.multi_hand_landmarks,
                        hand_res.multi_handedness):
        col  = HAND_R if hd.classification[0].label == "Left" else HAND_L
        spec = mp_draw.DrawingSpec(color=col, thickness=2, circle_radius=3)
        mp_draw.draw_landmarks(frame, lms, mp_hands_mod.HAND_CONNECTIONS, spec, spec)

def _draw_face_landmarks(frame, face_res):
    if not face_res.multi_face_landmarks:
        return
    mp_draw.draw_landmarks(
        frame,
        face_res.multi_face_landmarks[0],
        mp_face_mod.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(
            color=FACE_C, thickness=1, circle_radius=1),
    )


# ── Timer ring (top-right, 2-second hold) ─────────────────────────────────

def _draw_timer_ring(frame, progress):
    """Draw a circular countdown arc in the top-right corner."""
    h, w = frame.shape[:2]
    cx, cy = w - 50, 50
    radius = 30
    # background ring
    cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 3)
    # progress arc — starts at top (-90°), goes clockwise
    if progress > 0:
        angle_end = int(-90 + 360 * min(progress, 1.0))
        cv2.ellipse(frame, (cx, cy), (radius, radius),
                    0, -90, angle_end, GREEN, 3)
    # timer seconds text inside ring
    secs = progress * 2.0
    cv2.putText(frame, f"{secs:.1f}s", (cx - 18, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)


# ── Status bar (bottom) ───────────────────────────────────────────────────

def _draw_status(frame, hand_count, face_visible):
    h, w = frame.shape[:2]
    items = [
        (f"HANDS:{hand_count}", GREEN if hand_count > 0 else RED),
        ("FACE",                GREEN if face_visible    else RED),
    ]
    for i, (txt, col) in enumerate(items):
        x = w - 240 + i * 120
        cv2.rectangle(frame, (x, h - 38), (x + 110, h - 10), col, -1)
        cv2.putText(frame, txt, (x + 6, h - 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DARK, 2)


# ── Main detector class ────────────────────────────────────────────────────

class SignDetector:
    """
    Instantiated once inside generate_frames().
    Loads the model and mediapipe once, then processes frames.
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"model.h5 not found at {MODEL_PATH}. "
                "Run 2_train_model.py to generate it."
            )
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(
                f"labels.txt not found at {LABELS_PATH}."
            )

        self.model  = tf.keras.models.load_model(MODEL_PATH)
        self.labels = open(LABELS_PATH).read().splitlines()

        self.hands = mp_hands_mod.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_DETECT,
            min_tracking_confidence=MIN_TRACK,
        )
        self.face_mesh = mp_face_mod.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECT,
            min_tracking_confidence=MIN_TRACK,
        )

        self.votes = collections.deque(maxlen=VOTE_WINDOW)
        self.confs = collections.deque(maxlen=VOTE_WINDOW)

    def predict_frame(self, frame):
        """
        Run inference on one BGR frame.
        Returns (detected_label_str, confidence_float, annotated_frame).
        detected_label_str is "" if confidence < threshold.
        """
        h, w = frame.shape[:2]

        # IMPORTANT: do NOT flip — match collect_data.py which flips before saving
        # collect_data does cv2.flip(frame, 1) before extracting landmarks,
        # so we flip here too to keep the same orientation
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_res = self.hands.process(rgb)
        face_res = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        # Build feature vector and run model
        vec  = _build_feature_vector(hand_res, face_res, w, h)
        prob = self.model.predict(
            np.array([vec], dtype="float32"), verbose=0
        )[0]
        idx  = int(np.argmax(prob))
        conf = float(prob[idx])

        self.votes.append(idx)
        self.confs.append(conf)

        # Rolling majority vote (same as 3_interface.py)
        from collections import Counter
        best_idx  = Counter(self.votes).most_common(1)[0][0]
        best_conf = float(np.mean(
            [self.confs[i] for i, v in enumerate(self.votes) if v == best_idx]
        ))

        raw_label = self.labels[best_idx] if best_idx < len(self.labels) else ""
        if best_conf >= CONF_THRESHOLD:
            display_label = LABELS_DISPLAY_MAP.get(
                raw_label.lower(), raw_label.upper()
            )
        else:
            display_label = ""
            best_conf     = 0.0

        # Draw landmarks on frame
        _draw_face_landmarks(frame, face_res)
        _draw_hand_landmarks(frame, hand_res)

        # Status bar
        hand_count = (len(hand_res.multi_hand_landmarks)
                      if hand_res.multi_hand_landmarks else 0)
        _draw_status(frame, hand_count, bool(face_res.multi_face_landmarks))

        # Label + confidence top-left
        if display_label:
            conf_pct = int(best_conf * 100)
            cv2.putText(frame, f"{display_label}  {conf_pct}%",
                        (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

        return display_label, best_conf, frame

    def reset_votes(self):
        self.votes.clear()
        self.confs.clear()

    def close(self):
        self.hands.close()
        self.face_mesh.close()
