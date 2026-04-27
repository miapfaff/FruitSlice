"""central config values for gameplay and rendering.

all values are kept in one place so we can tweak difficulty/feel
without digging through multiple files.
"""

from pathlib import Path

# repo root (parent of `src/`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# mediapipe tasks hand model; downloaded on first run if missing.
HAND_LANDMARKER_MODEL_PATH = PROJECT_ROOT / "data" / "hand_landmarker.task"
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# window title shown by opencv.
WINDOW_NAME = "FruitSlice MVP"

# default camera device index (set to 1 for front camera; try 2 if 1 does not work).
CAMERA_DEVICE_INDEX = 1

# camera/frame render size.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# max active fruits allowed on screen.
MAX_FRUITS = 3
# hard cap for level-scaled active fruits.
MAX_FRUITS_CAP = 8
# chance that a spawn is a bomb instead of fruit.
BOMB_SPAWN_CHANCE = 0.16
# player health; bomb hits consume one life.
STARTING_LIVES = 3
# minimum time between spawns so the game does not flood.
FRUIT_SPAWN_COOLDOWN_SECONDS = 1.4
# lower bound for spawn cooldown as levels increase.
FRUIT_SPAWN_COOLDOWN_MIN_SECONDS = 0.45
# start radius (small = farther away look).
FRUIT_BASE_RADIUS = 24
# final radius (big = closer to player look).
FRUIT_MAX_RADIUS = 70
# upward launch speed range in pixels/sec.
FRUIT_MIN_LAUNCH_SPEED = 1100
FRUIT_MAX_LAUNCH_SPEED = 1300
# lateral speed range while fruit is airborne.
FRUIT_HORIZONTAL_SPEED = 180
# downward acceleration in pixels/sec^2.
FRUIT_GRAVITY = 1100
# time-based level progression.
LEVEL_UP_EVERY_SECONDS = 20.0
# scaling applied per level after level 1.
LEVEL_SPAWN_COOLDOWN_STEP = 0.08
LEVEL_SPEED_MULTIPLIER_STEP = 0.12
# short HUD banner duration when leveling up.
LEVEL_UP_BANNER_SECONDS = 1.0

# short cooldown avoids counting one swipe multiple times per frame burst.
SLICE_COOLDOWN_SECONDS = 0.08
# how many recent hand points to keep for drawing the trail.
SLICE_TRAIL_LENGTH = 8
# visual thickness of the hand slice trail line.
SLICE_LINE_THICKNESS = 4
