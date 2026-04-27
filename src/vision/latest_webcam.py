"""Threaded webcam reader that always exposes the freshest frame.

OpenCV may queue frames internally; slow per-frame work (e.g. MediaPipe) lets the
main loop fall behind and `read()` keeps returning stale images—often looking
fully frozen. A dedicated capture thread continuously pulls from the camera and
stores only the latest frame so the game always samples current video.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np


class LatestFrameCapture:
    """Single-consumer wrapper: one background thread calls `VideoCapture.read()`."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="webcam-capture", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running.is_set():
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._latest = frame

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Return a copy of the most recent frame (safe to draw on)."""
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
