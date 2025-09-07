import cv2

from .solver import detect_cube, solve_cube


def main():
    video = cv2.VideoCapture(0)
    try:
        cmd = detect_cube(video)
        if cmd == "detected":
            cmd = solve_cube(video)
        if cmd == "quit":
            print("Program finished due to keyboard command")
        elif cmd == "failed":
            print("Program finished due to unexpected error")
        else:
            print("Cube solved successfully")
    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
