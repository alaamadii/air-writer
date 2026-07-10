# Air Writing

Air Writing is a real-time computer-vision application that lets you draw in the air using hand gestures. It captures video from your webcam, detects hand landmarks with MediaPipe, and renders your movements on a virtual canvas with OpenCV.

## Features

- Draw using your index finger
- Move the cursor without drawing
- Erase parts of the canvas with a gesture
- Clear the entire canvas using a gesture or keyboard shortcut
- See live hand-landmark feedback
- Configure the camera, canvas size, brush, eraser, and detection confidence

## Requirements

- Python 3.10 or 3.11 recommended
- A working webcam
- Windows, macOS, or Linux with a desktop environment

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/alaamadii/air-writer.git
cd air-writer
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Start the application from the project directory:

```bash
python main.py
```

Allow camera access if your operating system asks for permission.

### Controls

| Input | Action |
| --- | --- |
| Index finger raised | Draw |
| Index and middle fingers raised | Move without drawing |
| Index, middle, and ring fingers raised | Erase |
| All five fingers raised | Clear the canvas |
| `C` | Clear the canvas |
| `Q` | Quit |

Gesture recognition can vary with camera angle and lighting. Keep your hand clearly visible and face your palm toward the camera for the most reliable results.

## Configuration

Edit `config.py` to change:

- Camera index and capture resolution
- Brush color and thickness
- Eraser thickness
- Hand detection and tracking confidence thresholds

Colors use OpenCV's BGR format rather than RGB.

## Project structure

```text
air-writer/
|-- main.py                 # Application loop and drawing behavior
|-- hand_tracker.py         # MediaPipe detection and gesture state
|-- config.py               # User-adjustable settings
|-- hand_landmarker.task    # MediaPipe hand-landmark model
|-- requirements.txt        # Python dependencies
|-- air-write-project.md    # Original project notes
`-- README.md
```

## Troubleshooting

- **Camera does not open:** close other applications using it, grant camera permission, or change `CAMERA_INDEX` in `config.py`.
- **No hand is detected:** improve the lighting, keep your whole hand in frame, and try lowering the confidence values in `config.py` slightly.
- **MediaPipe does not install:** use a supported 64-bit Python version; Python 3.10 or 3.11 generally provides the broadest compatibility.

## Privacy

Camera frames are processed locally and are not uploaded by this application.
