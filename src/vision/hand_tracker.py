from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandSlicePoint:
    """pixel-space point used by game slicing logic."""

    x: int
    y: int
    confidence: float


class HandTracker:
    """small wrapper around mediapipe hands for this game.

    this class hides mediapipe details and returns only what the game loop needs:
    the index finger tip point in pixel coordinates.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        # mp.solutions is the classic hands api.
        self._mp_hands = mp.solutions.hands
        # create the reusable detector/tracker object once.
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def get_index_tip(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[HandSlicePoint], Optional[object]]:
        """return the first hand's index-finger tip in pixel coordinates.

        returns:
        - HandSlicePoint + raw mediapipe results when a hand is found
        - None + raw results when no hand landmarks are present
        """
        # mediapipe expects rgb input, while opencv camera frames are bgr.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None, results

        # use only the first detected hand.
        hand_landmarks = results.multi_hand_landmarks[0]
        # select the index fingertip landmark from mediapipe's enum (enum is a list of constants)
        landmark = hand_landmarks.landmark[self._mp_hands.HandLandmark.INDEX_FINGER_TIP]
        # convert normalized coordinates (0..1) to pixel coordinates (x, y)
        h, w, _ = frame_bgr.shape
        point = HandSlicePoint(
            x=int(landmark.x * w),
            y=int(landmark.y * h),
            # visibility may be missing on some models, so default to 1.0.
            confidence=landmark.visibility if landmark.visibility else 1.0,
        )
        return point, results

    def close(self) -> None:
        # release mediapipe hands object resources
        self._hands.close()
