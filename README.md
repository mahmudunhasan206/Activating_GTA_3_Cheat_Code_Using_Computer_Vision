# GTA III Hand Gesture

Use a webcam to type GTA III's health cheat (`gesundheit`) when a closed fist
is detected. The app requires a stable open hand to arm, then triggers once
when the hand closes. The preview blurs the background, keeps the hand in
focus, outlines a recognized open palm or closed fist with a green box, and
uses larger labels for the detection. Open your hand again before the next
activation; a cooldown also prevents rapid retriggers.

## Requirements

- Windows
- Python 3.11 through 3.14
- A webcam
- An internet connection for the app's first run to download the hand-tracking
  model

## Install

From this folder in PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run the app with
`.venv\Scripts\python.exe` instead.
Replace `3.14` with your installed Python version if needed.

## Run

1. Start GTA III and leave it ready to receive keyboard input.
2. From this folder, run `python main.py`. On the first run, it downloads and
   caches the MediaPipe hand-tracking model in your local app-data folder.
3. Allow webcam access and show an open hand until the preview says it is armed.
4. Close your hand to type `gesundheit`. Open it again before triggering another
   cheat. The preview and console show an activation message when the closed
   palm triggers the health cheat.
5. Press **Q** in the camera preview to quit.

The cheat is typed into whichever window is focused, so focus GTA III before
making the gesture. Keep the camera preview visible while using the app.

Optional settings:

```powershell
python main.py --camera 1 --debounce 0.4 --cooldown 5
```

Run the gesture-state tests without installing webcam dependencies:

```powershell
python -m unittest -v
```
