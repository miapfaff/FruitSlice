"""central config values for gameplay and rendering.

all values are kept in one place so we can tweak difficulty/feel
without digging through multiple files.
"""

# window title shown by opencv.
WINDOW_NAME = "FruitSlice MVP"

# camera/frame render size.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# max active fruits allowed on screen.
MAX_FRUITS = 5
# minimum time between spawns so the game does not flood.
FRUIT_SPAWN_COOLDOWN_SECONDS = 0.7
# start radius (small = farther away look).
FRUIT_BASE_RADIUS = 24
# final radius (big = closer to player look).
FRUIT_MAX_RADIUS = 70
# movement speed range in pixels/sec.
FRUIT_MIN_SPEED = 240
FRUIT_MAX_SPEED = 380

# short cooldown avoids counting one swipe multiple times per frame burst.
SLICE_COOLDOWN_SECONDS = 0.08
# how many recent hand points to keep for drawing the trail.
SLICE_TRAIL_LENGTH = 8
# visual thickness of the hand slice trail line.
SLICE_LINE_THICKNESS = 4
