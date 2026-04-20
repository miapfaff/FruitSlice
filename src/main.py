from __future__ import annotations

from collections import deque

import cv2

from src import config
from src.game.game_state import GameState
from src.vision.hand_tracker import HandTracker


def draw_hud(frame, score: int, misses: int) -> None:
    """draw simple overlay text for core game stats and controls."""
    cv2.putText(frame, f"Score: {score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
    cv2.putText(frame, f"Misses: {misses}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
    cv2.putText(
        frame,
        "Press Q to quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
    )


def main() -> None:
    """entry point for webcam loop, game simulation, and rendering."""
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
    # keep a short history of hand points for both drawing and slice detection.
    trail: deque[tuple[int, int]] = deque(maxlen=config.SLICE_TRAIL_LENGTH)

    # baseline timestamp used for dt calculation.
    last_time = game.now()

    try:
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

            # update game objects.
            game.maybe_spawn_fruit(now)
            game.update(dt)

            # if we have at least two points, test newest segment for slicing.
            if len(trail) >= 2:
                p1 = trail[-2]
                p2 = trail[-1]
                game.try_slice(p1, p2, now)

            # draw all fruits.
            for fruit in game.fruits:
                fruit.draw(frame)

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

            # draw hud text and present frame.
            draw_hud(frame, game.score, game.misses)
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


if __name__ == "__main__":
    main()
