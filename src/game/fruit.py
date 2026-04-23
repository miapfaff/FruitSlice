from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import random

import cv2
import numpy as np

from src import config


FruitKind = Literal["orange", "watermelon", "apple"]


@dataclass
class Fruit:
    """single fruit used by the game simulation.

    we store position, velocity, radius growth, color, and sliced state.
    """

    x: float
    y: float
    vx: float
    vy: float
    kind: FruitKind
    radius: float
    target_radius: float
    sliced: bool = False

    def update(self, dt: float) -> None:
        """advance fruit position and growth by dt seconds."""
        # basic projectile motion with downward gravity.
        self.vy += config.FRUIT_GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        # increasing radius creates the visual illusion of moving toward camera
        growth_speed = 120.0
        self.radius = min(self.target_radius, self.radius + growth_speed * dt)

    def draw(self, frame: np.ndarray) -> None:
        """Draw stylized fruit with per-kind details."""
        _draw_fruit(frame, self.kind, self.x, self.y, self.radius)

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

    def touched_ground(self, height: int) -> bool:
        """return true when bottom of fruit touches screen bottom (ground)."""
        return (self.y + self.radius) >= height

    def out_of_bounds(self, width: int, height: int) -> bool:
        """remove fruit when it exits side/top padding to keep list bounded."""
        pad = 120
        return self.x < -pad or self.x > width + pad or self.y < -pad


@dataclass
class SlicedFruitHalf:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    kind: FruitKind
    side_sign: int  # -1 left half, +1 right half
    life_s: float = 0.45

    def update(self, dt: float) -> None:
        self.vy += config.FRUIT_GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life_s -= dt

    def draw(self, frame: np.ndarray) -> None:
        center = (int(self.x), int(self.y))
        axes = (max(4, int(self.radius * 0.8)), max(4, int(self.radius * 0.55)))
        start_angle = 90 if self.side_sign < 0 else -90
        end_angle = 270 if self.side_sign < 0 else 90
        color, flesh = _fruit_palette(self.kind)
        cv2.ellipse(frame, center, axes, 0, start_angle, end_angle, color, -1)
        cv2.ellipse(frame, center, axes, 0, start_angle, end_angle, (245, 245, 245), 1)
        # exposed interior
        inner_axes = (max(2, int(axes[0] * 0.72)), max(2, int(axes[1] * 0.72)))
        cv2.ellipse(frame, center, inner_axes, 0, start_angle, end_angle, flesh, -1)

    def expired(self) -> bool:
        return self.life_s <= 0.0


@dataclass
class JuiceParticle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color_bgr: tuple[int, int, int]
    life_s: float

    def update(self, dt: float) -> None:
        self.vy += config.FRUIT_GRAVITY * 0.75 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life_s -= dt

    def draw(self, frame: np.ndarray) -> None:
        if self.life_s <= 0:
            return
        cv2.circle(frame, (int(self.x), int(self.y)), max(1, int(self.radius)), self.color_bgr, -1)

    def expired(self) -> bool:
        return self.life_s <= 0.0


def make_slice_effects(fruit: Fruit) -> tuple[list[SlicedFruitHalf], list[JuiceParticle]]:
    halves = [
        SlicedFruitHalf(
            x=fruit.x - fruit.radius * 0.15,
            y=fruit.y,
            vx=fruit.vx - 220,
            vy=fruit.vy - 120,
            radius=fruit.radius,
            kind=fruit.kind,
            side_sign=-1,
        ),
        SlicedFruitHalf(
            x=fruit.x + fruit.radius * 0.15,
            y=fruit.y,
            vx=fruit.vx + 220,
            vy=fruit.vy - 120,
            radius=fruit.radius,
            kind=fruit.kind,
            side_sign=1,
        ),
    ]
    _, flesh = _fruit_palette(fruit.kind)
    particles: list[JuiceParticle] = []
    for _ in range(14):
        angle = random.uniform(0.0, math.tau)
        speed = random.uniform(120.0, 320.0)
        particles.append(
            JuiceParticle(
                x=fruit.x,
                y=fruit.y,
                vx=math.cos(angle) * speed + fruit.vx * 0.2,
                vy=math.sin(angle) * speed - 120.0,
                radius=random.uniform(1.5, 4.0),
                color_bgr=flesh,
                life_s=random.uniform(0.18, 0.42),
            )
        )
    return halves, particles


def _fruit_palette(kind: FruitKind) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if kind == "orange":
        return (40, 155, 245), (120, 190, 255)
    if kind == "watermelon":
        return (55, 180, 70), (90, 85, 220)
    return (45, 45, 215), (120, 170, 250)


def _draw_fruit(frame: np.ndarray, kind: FruitKind, x: float, y: float, radius: float) -> None:
    center = (int(x), int(y))
    skin, flesh = _fruit_palette(kind)
    r = int(radius)
    cv2.circle(frame, center, r, skin, -1)
    if kind == "watermelon":
        cv2.circle(frame, center, int(r * 0.85), (85, 230, 125), 3)
        cv2.circle(frame, center, int(r * 0.72), flesh, -1)
        # seeds
        for dx, dy in [(-0.25, -0.1), (0.2, -0.2), (-0.05, 0.15), (0.24, 0.14)]:
            px = int(x + dx * r)
            py = int(y + dy * r)
            cv2.ellipse(frame, (px, py), (2, 4), 0, 0, 360, (25, 25, 25), -1)
    elif kind == "orange":
        cv2.circle(frame, center, int(r * 0.9), skin, -1)
        # orange segment lines
        for ang in range(0, 180, 45):
            rad = math.radians(ang)
            dx = int(math.cos(rad) * r * 0.72)
            dy = int(math.sin(rad) * r * 0.72)
            cv2.line(frame, (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy), (120, 200, 255), 1)
    else:
        # apple: stem + highlight
        cv2.rectangle(
            frame,
            (center[0] - 2, center[1] - r - 8),
            (center[0] + 2, center[1] - r + 3),
            (30, 70, 120),
            -1,
        )
        cv2.ellipse(frame, (center[0] + 4, center[1] - r - 3), (4, 2), -20, 0, 360, (70, 170, 70), -1)
        cv2.circle(frame, (center[0] - int(r * 0.25), center[1] - int(r * 0.25)), max(2, int(r * 0.14)), (130, 180, 255), -1)
    cv2.circle(frame, center, r, (255, 255, 255), 2)
