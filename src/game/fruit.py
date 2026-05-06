"""fruit and bomb entities, slice debris, juice particles, and opencv drawing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import random

import cv2
import numpy as np

from src import config


FruitKind = Literal["orange", "watermelon", "apple", "bomb"]


@dataclass
class Fruit:
    """projectile with growing radius (pseudo-3d) and circle-vs-segment hit test."""

    x: float
    y: float
    vx: float
    vy: float
    kind: FruitKind
    radius: float
    target_radius: float
    sliced: bool = False

    def update(self, dt: float) -> None:
        """apply gravity, integrate position, grow radius toward `target_radius`."""
        # basic projectile motion with downward gravity.
        self.vy += config.FRUIT_GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        # increasing radius creates the visual illusion of moving toward camera
        growth_speed = 120.0
        self.radius = min(self.target_radius, self.radius + growth_speed * dt)

    def draw(self, frame: np.ndarray) -> None:
        """render skin/flesh details for orange, watermelon, apple, or bomb."""
        _draw_fruit(frame, self.kind, self.x, self.y, self.radius)

    def intersects_segment(self, p1: tuple[int, int], p2: tuple[int, int]) -> bool:
        """true if the closest point on segment p1–p2 to the center is within radius."""
        # use vectors so we can project the center onto the infinite line, then clamp t.
        center = np.array([self.x, self.y], dtype=float)
        p1_np = np.array(p1, dtype=float)
        p2_np = np.array(p2, dtype=float)
        seg = p2_np - p1_np
        seg_len_sq = float(np.dot(seg, seg))

        if seg_len_sq == 0:
            # degenerate segment: treat as a point swipe.
            distance = float(np.linalg.norm(center - p1_np))
            return distance <= self.radius

        # t in [0,1] picks the nearest point on the closed segment.
        t = float(np.dot(center - p1_np, seg) / seg_len_sq)
        t = max(0.0, min(1.0, t))
        closest = p1_np + t * seg
        # hit when distance from center to closest segment point <= circle radius.
        distance = float(np.linalg.norm(center - closest))
        return distance <= self.radius

    def touched_ground(self, height: int) -> bool:
        """true once falling (vy > 0) and lowest point reaches frame bottom."""
        return self.vy > 0 and (self.y + self.radius) >= height

    def out_of_bounds(self, width: int, height: int) -> bool:
        """true if fruit exits far left/right; top escape is allowed (arc may return)."""
        pad = 120
        # do not cull when y is above the frame; parabolas often peak off-screen.
        return self.x < -pad or self.x > width + pad


@dataclass
class SlicedFruitHalf:
    """short-lived half-ellipse chunk flying apart after a successful slice."""

    x: float
    y: float
    vx: float
    vy: float
    radius: float
    kind: FruitKind
    side_sign: int  # -1 left half, +1 right half
    life_s: float = 0.45

    def update(self, dt: float) -> None:
        """same gravity as fruit; decay `life_s` until `expired`."""
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
        # inner ellipse reads as exposed flesh.
        inner_axes = (max(2, int(axes[0] * 0.72)), max(2, int(axes[1] * 0.72)))
        cv2.ellipse(frame, center, inner_axes, 0, start_angle, end_angle, flesh, -1)

    def expired(self) -> bool:
        """true when lifetime elapsed; game loop should drop this half."""
        return self.life_s <= 0.0


@dataclass
class JuiceParticle:
    """small circle with gravity and fade-out; used for juice and bomb sparks."""
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color_bgr: tuple[int, int, int]
    life_s: float

    def update(self, dt: float) -> None:
        """lighter gravity than fruit; integrate and subtract from `life_s`."""
        self.vy += config.FRUIT_GRAVITY * 0.75 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life_s -= dt

    def draw(self, frame: np.ndarray) -> None:
        """skip drawing dead particles to avoid z-fighting on last frame."""
        if self.life_s <= 0:
            return
        cv2.circle(frame, (int(self.x), int(self.y)), max(1, int(self.radius)), self.color_bgr, -1)

    def expired(self) -> bool:
        """true when `life_s` exhausted."""
        return self.life_s <= 0.0


def make_slice_effects(fruit: Fruit) -> tuple[list[SlicedFruitHalf], list[JuiceParticle]]:
    """build visual feedback for one slice: halves + particles, or bomb-only burst."""
    if fruit.kind == "bomb":
        return [], make_bomb_explosion(fruit.x, fruit.y, fruit.vx, fruit.vy)

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
    # spray flesh-colored droplets in a ring, biased upward for readability.
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


def make_bomb_explosion(x: float, y: float, vx: float, vy: float) -> list[JuiceParticle]:
    """dense radial burst; warm colors + gray smoke; inherits some fruit velocity."""
    particles: list[JuiceParticle] = []
    for _ in range(36):
        angle = random.uniform(0.0, math.tau)
        speed = random.uniform(180.0, 460.0)
        # fiery orange/yellow sparks mixed with dark smoke.
        if random.random() < 0.7:
            color = random.choice([(20, 120, 255), (40, 180, 255), (70, 220, 255)])
        else:
            shade = random.randint(30, 80)
            color = (shade, shade, shade)
        particles.append(
            JuiceParticle(
                x=x,
                y=y,
                vx=math.cos(angle) * speed + vx * 0.25,
                vy=math.sin(angle) * speed + vy * 0.15,
                radius=random.uniform(2.0, 5.5),
                color_bgr=color,
                life_s=random.uniform(0.22, 0.6),
            )
        )
    return particles


def _fruit_palette(kind: FruitKind) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """return (skin_bgr, flesh_bgr) for drawing; bomb uses dark neutrals."""
    if kind == "orange":
        return (40, 155, 245), (120, 190, 255)
    if kind == "watermelon":
        return (55, 180, 70), (90, 85, 220)
    if kind == "bomb":
        return (45, 45, 45), (80, 80, 80)
    return (45, 45, 215), (120, 170, 250)


def _draw_fruit(frame: np.ndarray, kind: FruitKind, x: float, y: float, radius: float) -> None:
    """procedural opencv shapes per kind; shared white outline for visibility."""
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
    elif kind == "bomb":
        cv2.circle(frame, center, int(r * 0.95), (35, 35, 35), -1)
        cv2.circle(frame, center, int(r * 0.95), (120, 120, 120), 2)
        # fuse + spark
        cv2.line(
            frame,
            (center[0], center[1] - r),
            (center[0] + int(r * 0.4), center[1] - int(r * 1.5)),
            (90, 130, 160),
            2,
        )
        cv2.circle(frame, (center[0] + int(r * 0.45), center[1] - int(r * 1.55)), 3, (30, 200, 255), -1)
        cv2.putText(
            frame,
            "!",
            (center[0] - int(r * 0.2), center[1] + int(r * 0.25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, r / 35.0),
            (230, 230, 230),
            2,
        )
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
