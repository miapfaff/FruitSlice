from __future__ import annotations

import random
import time

from src import config
from src.game.fruit import Fruit


class GameState:
    """owns mutable game data and update rules.

    this keeps `main.py` focused on input/render loop while this class handles:
    - fruit spawning
    - per-frame simulation updates
    - slice scoring logic
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        # frame size is needed for spawn ranges and out-of-bounds checks.
        self.frame_width = frame_width
        self.frame_height = frame_height
        # active fruits currently alive in the scene.
        self.fruits: list[Fruit] = []
        # game counters shown in hud.
        self.score = 0
        self.misses = 0
        # timestamps for spawn/slice cooldown controls.
        self.last_spawn_time = 0.0
        self.last_slice_time = 0.0

    def maybe_spawn_fruit(self, now: float) -> None:
        """spawn a fruit when below limits and cooldown has elapsed."""
        if len(self.fruits) >= config.MAX_FRUITS:
            return
        if now - self.last_spawn_time < config.FRUIT_SPAWN_COOLDOWN_SECONDS:
            return

        # choose spawn side to vary incoming trajectories.
        side = random.choice(["left", "right", "bottom"])
        if side == "left":
            x, y = -30.0, random.uniform(120, self.frame_height - 120)
            vx = random.uniform(config.FRUIT_MIN_SPEED, config.FRUIT_MAX_SPEED)
            vy = random.uniform(-80, 80)
        elif side == "right":
            x, y = self.frame_width + 30.0, random.uniform(120, self.frame_height - 120)
            vx = -random.uniform(config.FRUIT_MIN_SPEED, config.FRUIT_MAX_SPEED)
            vy = random.uniform(-80, 80)
        else:
            x, y = random.uniform(120, self.frame_width - 120), self.frame_height + 30.0
            vx = random.uniform(-120, 120)
            vy = -random.uniform(config.FRUIT_MIN_SPEED, config.FRUIT_MAX_SPEED)

        # placeholder colors until sprite assets are added.
        color = random.choice(
            [
                (72, 120, 240),
                (78, 200, 98),
                (60, 220, 220),
                (170, 80, 210),
                (65, 180, 255),
            ]
        )
        fruit = Fruit(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            radius=config.FRUIT_BASE_RADIUS,
            target_radius=config.FRUIT_MAX_RADIUS,
            color_bgr=color,
        )
        self.fruits.append(fruit)
        self.last_spawn_time = now

    def update(self, dt: float) -> None:
        """step simulation and remove sliced/missed fruits."""
        # update all fruit motion/radius.
        for fruit in self.fruits:
            fruit.update(dt)

        alive: list[Fruit] = []
        for fruit in self.fruits:
            # sliced fruits are removed immediately in this mvp.
            if fruit.sliced:
                continue
            # fruit leaving play area increments miss counter.
            if fruit.out_of_bounds(self.frame_width, self.frame_height):
                self.misses += 1
                continue
            alive.append(fruit)
        self.fruits = alive

    def try_slice(self, p1: tuple[int, int], p2: tuple[int, int], now: float) -> int:
        """test swipe segment against all fruits and return count sliced."""
        # cooldown prevents overcounting one fast swipe over many consecutive frames.
        if now - self.last_slice_time < config.SLICE_COOLDOWN_SECONDS:
            return 0

        sliced_count = 0
        for fruit in self.fruits:
            if not fruit.sliced and fruit.intersects_segment(p1, p2):
                fruit.sliced = True
                sliced_count += 1

        if sliced_count > 0:
            # one point per fruit for now; easy to extend to combos later.
            self.score += sliced_count
            self.last_slice_time = now

        return sliced_count

    @staticmethod
    def now() -> float:
        # perf_counter is high-resolution and monotonic for frame timing.
        return time.perf_counter()
