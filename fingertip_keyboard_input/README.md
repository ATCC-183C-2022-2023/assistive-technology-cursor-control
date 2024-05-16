# Fingertip Keyboard Input
A computer application that allows keyboard input by drawing letters in the air, captured by a web camera.

## How to Run
From this directory, install packages using `pip install -r ../requirements.txt`
Within the `fingertip_keyboard_input` directory, run `python mouse_and_keyboard_control.py`

## Modes
There are two main modes: Mouse Mode and Keyboard Mode. To switch between modes, the user can use the “fist” gesture with both hands simultaneously.

### Mouse Mode Gesture Mapping
| Left   Hand       |  Right Hand | Command |
| ---------- | --------- | ------------ |
| 1      | 5          |  cursor moving (within bounding box)|
| 1      | Arrow      |  cursor: click|
| 1      | 2          |  cursor: double click|
| 1      | 1          |  cursor: right click|
| 2      | 1          |  scroll: up|
| 2      | Arrow      |  scroll: down|
| 2      | 2          |  scroll: left|
| 2      | 3          |  scroll: right|
| 3      | 1          |  volume: up|
| 3      | Arrow      |  volume: down|
| 3      | 2          |  volume: mute|
| 4      | 1          |  window: switch to previous app|
| 4      | 2          |  window: minimize active window|
| 4      | 3          |  browser: decrease text size|
| 4      | 4          |  browser: increase text size|
| 5      | 1          |  browser: new tab|
| 5      | 2          |  browser: address bar|
| 5      | 3          |  browser: close tab|
| 5      | Arrow      |  browser: switch between tabs|
| Fist   | Fist       |  Switch to Mouse Mode |

### Keyboard Mode Gesture Mapping
| Left Hand | Right Hand | Command | 
| ---------- | --------- | ------------ |
| n/a | 1 | Draw |
| n/a | Fist | Submit drawing |
| n/a | 4 | Clear drawing |
| 1 | n/a | Backspace |
| 4 | n/a | Switch between Alphabetical / Digit characters |
| Fist | Fist | Switch to Mouse Mode |

Note that the on-screen box activates based on the position of the center of the left palm.

---
## Further Work
- Top down implementation to reduce fatigue
- Multi-letter/entire word support
- User-createable custom gesture support
- More support for other special characters (currently supported: space, backspace)
- Resizeable windows
