import numpy as np

IMG_WIDTH = 640
IMG_HEIGHT = 360
IMG_CENTER = (IMG_WIDTH // 2, IMG_HEIGHT // 2)
COLORS = ["white", "yellow", "blue", "red", "green", "orange"]

# Original HSV ranges preserved
colour_ranges = {
    "blue": (np.array([70, 100, 220]), np.array([120, 255, 255])),
    "red": (np.array([110, 120, 150]), np.array([180, 255, 255])),
    "green": (np.array([45, 130, 60]), np.array([70, 255, 255])),
    "orange": (np.array([0, 175, 150]), np.array([25, 255, 255])),
    "yellow": (np.array([20, 130, 60]), np.array([50, 255, 255])),
    "white": (np.array([0, 0, 0]), np.array([180, 80, 255])),
}
