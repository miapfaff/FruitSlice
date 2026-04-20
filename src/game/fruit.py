from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Fruit:
    """single fruit used by the game simulation.

    we store position, velocity, radius growth, color, and sliced state.
    """

    x: float
    y: float
    vx: float
    vy: float
    radius: float
    target_radius: float
    color_bgr: tuple[int, int, int]
    sliced: bool = False

    def update(self, dt: float) -> None:
        """advance fruit position and growth by dt seconds."""
        # basic linear motion
        self.x += self.vx * dt
        self.y += self.vy * dt
        # increasing radius creates the visual illusion of moving toward camera
        growth_speed = 120.0
        self.radius = min(self.target_radius, self.radius + growth_speed * dt)

    def draw(self, frame: np.ndarray) -> None:
        """draw filled fruit plus thin outline for visibility."""
        center = (int(self.x), int(self.y))
        cv2.circle(frame, center, int(self.radius), self.color_bgr, -1)
        cv2.circle(frame, center, int(self.radius), (255, 255, 255), 2)

    def intersects_segment(self, p1: tuple[int, int], p2: tuple[int, int]) -> bool:
        """return true if a swipe segment passes through the fruit circle."""
        # represent all points as float vectors for projection math
        center = np.array([self.x, self.y], dtype=float)
        p1_np = np.array(p1, dtype=float)
        p2_np = np.array(p2, dtype=float)
        seg = p2_np - p1_np
        seg_len_sq = float(np.dot(seg, seg))

        if seg_len_sq == 0:
            # if there is no movement, just check point-to-center distance
            distance = float(np.linalg.norm(center - p1_np))
            return distance <= self.radius

        # project center onto segment and clamp to segment bounds [0, 1]
        t = float(np.dot(center - p1_np, seg) / seg_len_sq)
        t = max(0.0, min(1.0, t))
        closest = p1_np + t * seg
        # slice hit if closest point on segment lies inside fruit radius.
        distance = float(np.linalg.norm(center - closest))
        return distance <= self.radius

    def out_of_bounds(self, width: int, height: int) -> bool:
        """treat fruit as missed when it leaves padded screen bounds."""
        pad = 100
        return self.x < -pad or self.x > width + pad or self.y < -pad or self.y > height + pad
