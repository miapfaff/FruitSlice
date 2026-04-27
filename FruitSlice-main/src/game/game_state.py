from __future__ import annotations

import random
import time

from src import config
from src.game.fruit import Fruit, JuiceParticle, SlicedFruitHalf, make_slice_effects


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
        self.sliced_halves: list[SlicedFruitHalf] = []
        self.juice_particles: list[JuiceParticle] = []
        # game counters shown in hud.
        self.score = 0
        self.misses = 0
        self.lives = config.STARTING_LIVES
        self.game_over = False
        # timestamps for spawn/slice cooldown controls.
        self.last_spawn_time = 0.0
        self.last_slice_time = 0.0
        self.run_started_at = 0.0
        self.elapsed_time = 0.0
        self.level = 1
        self.last_level_up_at = -999.0

    def refresh_progression(self, now: float) -> None:
        """update elapsed time + level from current run clock."""
        if self.game_over:
            return
        self.elapsed_time = max(0.0, now - self.run_started_at)
        next_level = 1 + int(self.elapsed_time // config.LEVEL_UP_EVERY_SECONDS)
        if next_level > self.level:
            self.level = next_level
            self.last_level_up_at = now
        else:
            self.level = next_level

    def maybe_spawn_fruit(self, now: float) -> None:
        """spawn a fruit when below limits and cooldown has elapsed."""
        if self.game_over:
            return
        max_fruits = min(config.MAX_FRUITS_CAP, config.MAX_FRUITS + max(0, self.level - 1))
        if len(self.fruits) >= max_fruits:
            return
        cooldown = max(
            config.FRUIT_SPAWN_COOLDOWN_MIN_SECONDS,
            config.FRUIT_SPAWN_COOLDOWN_SECONDS
            - config.LEVEL_SPAWN_COOLDOWN_STEP * max(0, self.level - 1),
        )
        if now - self.last_spawn_time < cooldown:
            return

        # all fruits launch upward from the bottom, then fall due to gravity.
        speed_mult = 1.0 + config.LEVEL_SPEED_MULTIPLIER_STEP * max(0, self.level - 1)
        x = random.uniform(120, self.frame_width - 120)
        y = self.frame_height + 30.0
        vx = random.uniform(-config.FRUIT_HORIZONTAL_SPEED, config.FRUIT_HORIZONTAL_SPEED) * speed_mult
        vy = -random.uniform(config.FRUIT_MIN_LAUNCH_SPEED, config.FRUIT_MAX_LAUNCH_SPEED) * speed_mult
        if random.random() < config.BOMB_SPAWN_CHANCE:
            kind = "bomb"
        else:
            kind = random.choice(["orange", "watermelon", "apple"])
        fruit = Fruit(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            kind=kind,
            radius=config.FRUIT_BASE_RADIUS,
            target_radius=config.FRUIT_MAX_RADIUS,
        )
        self.fruits.append(fruit)
        self.last_spawn_time = now

    def update(self, dt: float) -> None:
        """step simulation and remove sliced/missed fruits."""
        if self.game_over:
            for half in self.sliced_halves:
                half.update(dt)
            for particle in self.juice_particles:
                particle.update(dt)
            self.sliced_halves = [half for half in self.sliced_halves if not half.expired()]
            self.juice_particles = [p for p in self.juice_particles if not p.expired()]
            return

        # update all fruit motion/radius.
        for fruit in self.fruits:
            fruit.update(dt)
        for half in self.sliced_halves:
            half.update(dt)
        for particle in self.juice_particles:
            particle.update(dt)

        alive: list[Fruit] = []
        for fruit in self.fruits:
            # sliced fruits are removed immediately in this mvp.
            if fruit.sliced:
                continue
            # missing means fruit hits ground before the player slices it.
            if fruit.touched_ground(self.frame_height):
                self.misses += 1
                continue
            # clean up impossible trajectories that leave play area.
            if fruit.out_of_bounds(self.frame_width, self.frame_height):
                continue
            alive.append(fruit)
        self.fruits = alive
        self.sliced_halves = [half for half in self.sliced_halves if not half.expired()]
        self.juice_particles = [p for p in self.juice_particles if not p.expired()]

    def try_slice(self, p1: tuple[int, int], p2: tuple[int, int], now: float) -> int:
        """test swipe segment against all fruits and return count sliced."""
        if self.game_over:
            return 0
        # cooldown prevents overcounting one fast swipe over many consecutive frames.
        if now - self.last_slice_time < config.SLICE_COOLDOWN_SECONDS:
            return 0

        sliced_count = 0
        for fruit in self.fruits:
            if not fruit.sliced and fruit.intersects_segment(p1, p2):
                fruit.sliced = True
                halves, particles = make_slice_effects(fruit)
                self.sliced_halves.extend(halves)
                self.juice_particles.extend(particles)
                if fruit.kind == "bomb":
                    self.lives -= 1
                    if self.lives <= 0:
                        self.lives = 0
                        self.game_over = True
                else:
                    sliced_count += 1

        if sliced_count > 0:
            # one point per fruit for now; easy to extend to combos later.
            self.score += sliced_count
            self.last_slice_time = now

        return sliced_count

    def reset(self) -> None:
        self.fruits.clear()
        self.sliced_halves.clear()
        self.juice_particles.clear()
        self.score = 0
        self.misses = 0
        self.lives = config.STARTING_LIVES
        self.game_over = False
        self.last_spawn_time = 0.0
        self.last_slice_time = 0.0
        self.run_started_at = self.now()
        self.elapsed_time = 0.0
        self.level = 1
        self.last_level_up_at = -999.0

    @staticmethod
    def now() -> float:
        # perf_counter is high-resolution and monotonic for frame timing.
        return time.perf_counter()
