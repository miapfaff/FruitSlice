"""application entry: webcam capture, mediapipe hand trail, pygame audio, opencv ui.

the main loop mirrors the camera for natural movement, maintains a short deque of
index-finger positions for slice segments, and drives `GameState` when playing.
start / game-over screens use opencv mouse callbacks for clickable buttons.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import pygame

from src import config
from src.game.game_state import GameState
from src.vision.hand_tracker import HandTracker


@dataclass
class UiState:
    """minimal ui state shared with opencv mouse callback via `param`."""

    mode: str = "start"  # start | playing | game_over
    mouse_x: int = 0
    mouse_y: int = 0
    clicked: bool = False


def draw_hud(frame, score: int, misses: int, lives: int, level: int) -> None:
    """draw overlay text for score, misses, lives, level, and quit hint."""
    cv2.putText(frame, f"Score: {score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
    cv2.putText(frame, f"Misses: {misses}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
    cv2.putText(
        frame,
        f"Lives: {lives}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (120, 220, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Level: {level}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (130, 235, 130),
        2,
    )
    cv2.putText(
        frame,
        "Press Q to quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
    )


def _button_rect(frame: np.ndarray) -> tuple[int, int, int, int]:
    """return (x1, y1, x2, y2) for the centered start/play-again button."""
    h, w, _ = frame.shape
    bw, bh = 260, 74
    x1 = (w - bw) // 2
    y1 = int(h * 0.62)
    return x1, y1, x1 + bw, y1 + bh


def _draw_button(frame: np.ndarray, text: str, hovered: bool) -> tuple[int, int, int, int]:
    """fill rounded-rectangle style button; brighter green when `hovered`. returns rect."""
    x1, y1, x2, y2 = _button_rect(frame)
    color = (75, 190, 90) if hovered else (55, 150, 70)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 235, 235), 2)
    cv2.putText(
        frame,
        text,
        (x1 + 38, y1 + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (245, 245, 245),
        2,
    )
    return x1, y1, x2, y2


def _in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    """true if pixel (x, y) lies inside axis-aligned rect."""
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _on_mouse(event: int, x: int, y: int, _flags: int, param: UiState) -> None:
    """opencv callback: track cursor; set `clicked` on left button down."""
    param.mouse_x = x
    param.mouse_y = y
    if event == cv2.EVENT_LBUTTONDOWN:
        param.clicked = True


def main() -> None:
    """run until user presses q: read frames, update game, draw, handle ui modes."""
    # open default webcam (device index 0).
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # request target frame size from config.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # create hand tracker + game state.
    tracker = HandTracker(max_num_hands=1)
    game = GameState(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    ui = UiState()
    # keep a short history of hand points for both drawing and slice detection.
    trail: deque[tuple[int, int]] = deque(maxlen=config.SLICE_TRAIL_LENGTH)

    # baseline timestamp used for dt calculation.
    last_time = game.now()
    sound_dir = config.PROJECT_ROOT / "assets" / "sounds"
    pygame.mixer.init()
    slice_sound = pygame.mixer.Sound(str(sound_dir / "slice.wav"))
    explosion_sound = pygame.mixer.Sound(str(sound_dir / "explosion.wav"))
    bg_music = sound_dir / "background.mp3"
    try:
        pygame.mixer.music.load(str(bg_music))
        pygame.mixer.music.set_volume(0.55)
        pygame.mixer.music.play(loops=-1)
    except pygame.error as exc:
        print(f"Background music could not start ({bg_music}): {exc}", file=sys.stderr)

    try:
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(config.WINDOW_NAME, _on_mouse, ui)
        while True:
            # read one frame from webcam.
            ok, frame = cap.read()
            if not ok:
                continue

            # mirror horizontally so movement feels natural to player.
            frame = cv2.flip(frame, 1)
            # dt (delta time) keeps movement frame-rate independent.
            now = game.now()
            dt = max(1.0 / 120.0, now - last_time)
            last_time = now

            # get fingertip location from current frame.
            point, _results = tracker.get_index_tip(frame)
            if point is not None:
                # append latest point so swipe can be treated as a segment.
                trail.append((point.x, point.y))
            else:
                # if hand is lost, clear trail so we do not connect stale points.
                trail.clear()

            if ui.mode == "playing":
                # advance level timer, spawn rules, and physics.
                game.refresh_progression(now)
                game.maybe_spawn_fruit(now)
                game.update(dt)

                # slice uses the last motion segment only (previous tip -> current tip).
                if len(trail) >= 2:
                    p1 = trail[-2]
                    p2 = trail[-1]
                    # pre-check bombs so we play explosion even when try_slice returns 0 fruit.
                    bomb_will_be_hit = any(
                        (not fruit.sliced)
                        and fruit.kind == "bomb"
                        and fruit.intersects_segment(p1, p2)
                        for fruit in game.fruits
                    )
                    sliced_count = game.try_slice(p1, p2, now)
                    if bomb_will_be_hit:
                        explosion_sound.play()
                    elif sliced_count > 0:
                        slice_sound.play()
                if game.game_over:
                    ui.mode = "game_over"
            else:
                # start / game_over: still advance dt-based effects (halves, particles).
                game.update(dt)

            # draw all fruits.
            for fruit in game.fruits:
                fruit.draw(frame)
            for half in game.sliced_halves:
                half.draw(frame)
            for particle in game.juice_particles:
                particle.draw(frame)

            # draw the hand trail as a polyline.
            if len(trail) >= 2:
                for i in range(1, len(trail)):
                    cv2.line(
                        frame,
                        trail[i - 1],
                        trail[i],
                        (255, 255, 255),
                        config.SLICE_LINE_THICKNESS,
                    )

            if ui.mode == "start":
                # dim live feed; title + button; click inside rect starts run.
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
                frame = cv2.addWeighted(overlay, 0.42, frame, 0.58, 0.0)
                cv2.putText(frame, "FruitSlice", (frame.shape[1] // 2 - 160, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                cv2.putText(frame, "Slice fruit. Avoid bombs.", (frame.shape[1] // 2 - 190, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
                rect = _draw_button(
                    frame,
                    "Start Game",
                    _in_rect(ui.mouse_x, ui.mouse_y, _button_rect(frame)),
                )
                if ui.clicked and _in_rect(ui.mouse_x, ui.mouse_y, rect):
                    game.reset()
                    trail.clear()
                    ui.mode = "playing"
                ui.clicked = False
            elif ui.mode == "game_over":
                # stronger dim; final score; play again resets state and trail.
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (25, 25, 25), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0.0)
                cv2.putText(frame, "Game Over", (frame.shape[1] // 2 - 170, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (80, 80, 245), 4)
                cv2.putText(frame, f"Final Score: {game.score}", (frame.shape[1] // 2 - 140, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
                rect = _draw_button(
                    frame,
                    "Play Again",
                    _in_rect(ui.mouse_x, ui.mouse_y, _button_rect(frame)),
                )
                if ui.clicked and _in_rect(ui.mouse_x, ui.mouse_y, rect):
                    game.reset()
                    trail.clear()
                    ui.mode = "playing"
                ui.clicked = False

            if (
                ui.mode == "playing"
                and now - game.last_level_up_at <= config.LEVEL_UP_BANNER_SECONDS
            ):
                cv2.putText(
                    frame,
                    f"Level Up! {game.level}",
                    (frame.shape[1] // 2 - 145, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (120, 240, 255),
                    3,
                )

            # draw hud text and present frame.
            draw_hud(frame, game.score, game.misses, game.lives, game.level)
            cv2.imshow(config.WINDOW_NAME, frame)

            # keyboard handling (q to quit).
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        # always release resources, even if an exception occurs.
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.music.stop()
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
