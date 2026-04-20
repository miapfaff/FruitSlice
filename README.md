# FruitSlice

Computer vision fruit-slicing game prototype using Python, OpenCV, and MediaPipe.  
Players slice flying fruit by moving their hand across the webcam feed.

## current features

- Webcam input with real-time hand tracking (MediaPipe Hands)
- Index finger trail used as the slice path
- objects spawn from sides/bottom and "fly toward" player by growing in size
- Slice collision checks against the hand movement segment
- Score + miss counters

## structure/layout so far

```text
FruitSlice/
  src/
    config.py
    main.py
    game/
      game_state.py
      fruit.py
    vision/
      hand_tracker.py
  requirements.txt
```

## after pulling:

python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --upgrade --force-reinstall -r requirements.txt
python -m src.main

Press `Q` to quit.

## to do

- Replace circle fruits with sprite images
- Add sliced-fruit animation/effects and audio
- Add start/game-over screens
- Add combo system and difficulty scaling
