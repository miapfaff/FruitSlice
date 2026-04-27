from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src import config


@dataclass
class HandSlicePoint:
    """pixel-space point used by game slicing logic."""

    x: int
    y: int
    confidence: float


class HandTracker:
    """Small wrapper around MediaPipe hand tracker for this game."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        _ensure_hand_landmarker_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(config.HAND_LANDMARKER_MODEL_PATH),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._tip_index = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP

    def get_index_tip(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[HandSlicePoint], Optional[object]]:
        """Return the first hand index-finger tip in pixel coordinates."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if not frame_rgb.flags["C_CONTIGUOUS"]:
            frame_rgb = np.ascontiguousarray(frame_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return None, results

        landmark = results.hand_landmarks[0][self._tip_index]
        if landmark.x is None or landmark.y is None:
            return None, results

        h, w, _ = frame_bgr.shape
        point = HandSlicePoint(
            x=int(landmark.x * w),
            y=int(landmark.y * h),
            confidence=landmark.visibility if landmark.visibility is not None else 1.0,
        )
        return point, results

    def close(self) -> None:
        self._landmarker.close()


def _ensure_hand_landmarker_model() -> None:
    path = config.HAND_LANDMARKER_MODEL_PATH
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(config.HAND_LANDMARKER_MODEL_URL, timeout=120) as resp:
            path.write_bytes(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not download the MediaPipe hand landmarker model. "
            f"Download it manually from {config.HAND_LANDMARKER_MODEL_URL} "
            f"and save it as {path}"
        ) from exc
