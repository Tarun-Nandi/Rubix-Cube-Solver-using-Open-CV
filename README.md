# Rubik's Cube Solver using OpenCV

## About The Project
The application leverages your webcam to identify the distinct colours on each face of the cube. Then it utilizes augmented reality to showcase the necessary moves to solve the scrambled cube. After each move, the subsequent move is accurately displayed on the corresponding side of the cube. Additionally, a 2D representation of the cube's scrambled state is presented after scanning each face, aiding users in comprehending the current state of the cube. Ultimately, the project aims to enable any user to solve the cube swiftly, even without prior knowledge of its notation.

## Project Structure
The project has been refactored into a modular package structure for better maintainability:

```
rubiks_cv/
├── __init__.py          # Package initialization
├── config.py            # Constants and HSV color ranges
├── cube.py              # Cube class and rotation logic
├── utils.py             # Utility functions
├── vision.py            # Computer vision functions
├── ui.py                # UI and drawing functions
├── solver.py            # Solving logic and orchestration
└── main.py              # Entry point
```

## How the Cube-Solver works
1) The application first prompts the user to show a certain colour-centred face to the camera where the colours of each of the smaller facelets that make up the face are recorded. This is done using a range of masks that filter out a specific colour that is within a predefined set of HSV values.

    ![image](https://github.com/user-attachments/assets/fa05e602-4351-4370-8143-88e7ab4730a7)
  
2) Once the user has correctly shown all 6 faces of the cube to the webcam, the state of the cube has been recorded and with the help of the kociemba library the moves required to solve the scrambled cube are calculated.
3) These moves are then displayed on the cube one by one and once the user has followed all the displayed instructions the Rubik's cube would have been solved.

   ![image](https://github.com/user-attachments/assets/e46700a0-ab57-46b5-80ad-ab2012c1010c)  ![image](https://github.com/user-attachments/assets/10412065-f1ab-49f8-8793-efe524162e96)

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Webcam/camera access
- A Rubik's cube

### Installation
1) Clone or download this repository
2) Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python numpy kociemba
```

### Running the Application

#### Option 1: Run the modular version (recommended)
```bash
python -m rubiks_cv.main
```

#### Option 2: Run the original single-file version
```bash
python Main.py
```

### Usage Instructions

1. **Launch the application** - The camera window will open
2. **Detection Phase**:
   - Follow the on-screen instructions to show each face of the cube
   - The program will ask you to show faces in this order: white, yellow, blue, red, green, orange
   - Position the cube so the specified center color is facing the camera
   - Press 'y' to confirm detection or 'n' to restart
3. **Solving Phase**:
   - Follow the visual arrows displayed on the cube
   - Execute each move as shown
   - Continue until the cube is solved

### Controls
- **'q'** - Quit the program
- **'y'** - Confirm face detection
- **'n'** - Restart face detection

## Important Note: HSV Color Ranges

⚠️ **The HSV color ranges in `rubiks_cv/config.py` are calibrated for specific lighting conditions and cube colors. You may need to adjust these ranges for your environment.**

The current HSV ranges are:
```python
colour_ranges = {
    "blue": (np.array([70, 100, 220]), np.array([120, 255, 255])),
    "red": (np.array([110, 120, 150]), np.array([180, 255, 255])),
    "green": (np.array([45, 130, 60]), np.array([70, 255, 255])),
    "orange": (np.array([0, 175, 150]), np.array([25, 255, 255])),
    "yellow": (np.array([20, 130, 60]), np.array([50, 255, 255])),
    "white": (np.array([0, 0, 0]), np.array([180, 80, 255])),
}
```

**Future Improvement**: Automatic HSV range calibration is a planned feature to make the solver work reliably across different lighting conditions and cube types.

## Troubleshooting

- **Camera not working**: Ensure your webcam is not being used by another application
- **Poor color detection**: Adjust the HSV ranges in `config.py` for your lighting conditions
- **Qt platform plugin error**: This is a common OpenCV issue on Linux and doesn't affect functionality
