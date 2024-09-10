import cv2
import kociemba
import numpy as np
from math import sin, cos, pi
from collections import OrderedDict

IMG_WIDTH = 640
IMG_HEIGHT = 360
IMG_CENTER = (IMG_WIDTH // 2, IMG_HEIGHT // 2)
COLORS = ["white", "yellow", "blue", "red", "green", "orange"]
colour_ranges = {'blue': (np.array([70, 100, 220]), np.array([120, 255, 255])),
                 'red': (np.array([110, 120, 150]), np.array([180, 255, 255])),
                 'green': (np.array([45, 130, 60]), np.array([70, 255, 255])),
                 'orange': (np.array([0, 175, 150]), np.array([25, 255, 255])),
                 'yellow': (np.array([20, 130, 60]), np.array([50, 255, 255])),
                 'white': (np.array([0, 0, 0]), np.array([180, 80, 255]))
                 }


def color_to_letter(color):
    # A dictionary to map color name to a specific letter.
    color_letter_map = {
        "blue": "F",
        "green": "B",
        "white": "U",
        "yellow": "D",
        "red": "L",
        "orange": "R"
    }
    # If the given color is listed in the dictionary, the method returns the corresponding letter.
    if color in color_letter_map:
        return color_letter_map[color]
    else:
        # If the color is not listed, it will raise a ValueError indicating the input color is invalid.
        raise ValueError(f"Invalid color: {color}")


def parse_turn(turn_str):
    # Initialize a flag to determine whether it's a counter-clockwise turn
    is_prime = False
    # Validate input: turn string should be either 2 characters or 1 character.
    if len(turn_str) > 2 or len(turn_str) == 0:
        raise ValueError(f"Invalid turn: {turn_str}")

    if len(turn_str) == 2:
        if turn_str[1] == "'":
            is_prime = True
            turn = turn_str[0]
            # If the second character is  2, it means turn the face twice (180 degrees) in clockwise
            # direction
        elif turn_str[1] == "2":
            turn = turn_str[0] * 2
        else:
            raise ValueError(f"Invalid turn: {turn_str}")
    else:
        turn = turn_str

    return turn, is_prime


class Cube:
    # Initialize the Cube class
    def __init__(self):
        empty_face = np.empty([3, 3], '<U6')
        # Create an empty 3x3 numpy array to represent an empty face of the cube
        self.state = {"white": empty_face, "orange": empty_face, "blue": empty_face,
                      "yellow": empty_face, "red": empty_face, "green": empty_face}
        # Create an empty 3x3 numpy array to represent an empty face of the cube

    def save_face(self, color, face):
        # The color is the key for the face in the state dictionary
        self.state[color] = face

    def get_face(self, color):
        # color is the key for the face in state, return the corresponding face
        return self.state[color]

    def get_solution(self):
        # Initialize an empty string to store the cube's current state
        state_str = ''
        for color in self.state:
            # Get 2D array representation of current face
            face = self.state[color]
            for i in range(3):
                for j in range(3):
                    # Append the letter representation of the color of the cell to the state string
                    state_str += color_to_letter(face[i][j])
        solution = kociemba.solve(state_str)
        # Use the Kociemba's algorithm to solve the cube based on the current state
        # and return the solution as a sequence of moves
        return solution

    def call_turn(self, turn_str):
        # Parse the turn string to get the face to turn and whether it's a prime turn
        turn, is_prime = parse_turn(turn_str)
        # Mapping of turn characters to the corresponding rotation methods
        turn_methods = {
            "R": lambda: self.rotate_right_face(is_prime),
            "L": lambda: self.rotate_left_face(is_prime),
            "U": lambda: self.rotate_up_face(is_prime),
            "D": lambda: self.rotate_down_face(is_prime),
            "F": lambda: self.rotate_front_face(is_prime),
            "B": lambda: self.rotate_back_face(is_prime),
        }
        if turn not in turn_methods:
            raise ValueError(f"Invalid turn: {turn_str}")
        turn_methods[turn]()

    def rotate_right_face(self, is_prime):
        # Make a copy of current faces that will be affected by the move
        current_blue = self.state["blue"].copy()
        current_white = self.state["white"].copy()
        current_green = self.state["green"].copy()
        current_yellow = self.state["yellow"].copy()
        updated_blue = current_blue.copy()
        updated_white = current_white.copy()
        updated_green = current_green.copy()
        updated_yellow = current_yellow.copy()

        # Iterate to change the cell colors on the edge
        # The right edge of the blue face is moved to the yellow face
        # The right edge of the white face is moved to the blue face
        # The left edge of the green face is moved to the white face
        # The right edge of the yellow face is moved to the green face
        for row in range(3):
            row_reversed = 2 - row
            if is_prime:
                updated_yellow[row][2] = current_blue[row][2]
                updated_blue[row][2] = current_white[row][2]
                updated_white[row][2] = current_green[row_reversed][0]
                updated_green[row][0] = current_yellow[row_reversed][2]
            else:
                updated_blue[row][2] = current_yellow[row][2]
                updated_white[row][2] = current_blue[row][2]
                updated_green[row][0] = current_white[row_reversed][2]
                updated_yellow[row][2] = current_green[row_reversed][0]

        # Rotate the orange face
        current_orange = self.state["orange"].copy()
        updated_orange = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_orange[row][col] = current_orange[col][2 - row]
                else:
                    updated_orange[row][col] = current_orange[2 - col][row]

        # Save the updated faces back to the cube's state
        self.save_face("blue", updated_blue)
        self.save_face("white", updated_white)
        self.save_face("green", updated_green)
        self.save_face("yellow", updated_yellow)
        self.save_face("orange", updated_orange)

    def rotate_left_face(self, is_prime):
        # Make a copy of current faces that will be affected by the move
        current_blue = self.state["blue"].copy()
        current_white = self.state["white"].copy()
        current_green = self.state["green"].copy()
        current_yellow = self.state["yellow"].copy()
        updated_blue = current_blue.copy()
        updated_white = current_white.copy()
        updated_green = current_green.copy()
        updated_yellow = current_yellow.copy()

        # Iterate to change the cell colors on the edge
        # The left edge of the blue face is moved to the yellow face
        # The left edge of the white face is moved to the blue face
        # The right edge of the green face is moved to the white face
        # The left edge of the yellow face is moved to the green face
        for row in range(3):
            row_reversed = 2 - row
            if is_prime:
                updated_blue[row][0] = current_yellow[row][0]
                updated_white[row][0] = current_blue[row][0]
                updated_green[row][2] = current_white[row_reversed][0]
                updated_yellow[row][0] = current_green[row_reversed][2]
            else:
                updated_yellow[row][0] = current_blue[row][0]
                updated_blue[row][0] = current_white[row][0]
                updated_white[row][0] = current_green[row_reversed][2]
                updated_green[row][2] = current_yellow[row_reversed][0]

        # Rotate the red face
        current_red = self.state["red"].copy()
        updated_red = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_red[row][col] = current_red[col][2 - row]
                else:
                    updated_red[row][col] = current_red[2 - col][row]

        # Save the updated faces back to the cube's state
        self.save_face("blue", updated_blue)
        self.save_face("white", updated_white)
        self.save_face("green", updated_green)
        self.save_face("yellow", updated_yellow)
        self.save_face("red", updated_red)

    def rotate_up_face(self, is_prime):
        # Make a copy of current faces that will be affected by the move
        current_blue = self.state["blue"].copy()
        current_red = self.state["red"].copy()
        current_green = self.state["green"].copy()
        current_orange = self.state["orange"].copy()
        updated_blue = current_blue.copy()
        updated_red = current_red.copy()
        updated_green = current_green.copy()
        updated_orange = current_orange.copy()

        # Iterate to change the cell colors on the edge
        # The top edge of the orange face is moved to the blue face
        # The top edge of the blue face is moved to the red face
        # The top edge of the red face is moved to the green face
        # The top edge of the green face is moved to the orange face
        for col in range(3):
            if is_prime:
                updated_orange[0][col] = current_blue[0][col]
                updated_blue[0][col] = current_red[0][col]
                updated_red[0][col] = current_green[0][col]
                updated_green[0][col] = current_orange[0][col]
            else:
                updated_blue[0][col] = current_orange[0][col]
                updated_red[0][col] = current_blue[0][col]
                updated_green[0][col] = current_red[0][col]
                updated_orange[0][col] = current_green[0][col]

        # Rotate the white face
        current_white = self.state["white"].copy()
        updated_white = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_white[row][col] = current_white[col][2 - row]
                else:
                    updated_white[row][col] = current_white[2 - col][row]

        # Save the updated faces back to the cube's state
        self.save_face("blue", updated_blue)
        self.save_face("red", updated_red)
        self.save_face("green", updated_green)
        self.save_face("orange", updated_orange)
        self.save_face("white", updated_white)

    def rotate_down_face(self, is_prime):
        # Make a copy of current faces that will be affected by the move
        current_blue = self.state["blue"].copy()
        current_red = self.state["red"].copy()
        current_green = self.state["green"].copy()
        current_orange = self.state["orange"].copy()
        updated_blue = current_blue.copy()
        updated_red = current_red.copy()
        updated_green = current_green.copy()
        updated_orange = current_orange.copy()

        # Iterate to change the cell colors on the edge
        # The bottom edge of the blue face is moved to the orange face
        # The bottom edge of the red face is moved to the blue face
        # The bottom edge of the green face is moved to the red face
        # The bottom edge of the orange face is moved to the green face
        for col in range(3):
            if is_prime:
                updated_blue[2][col] = current_orange[2][col]
                updated_red[2][col] = current_blue[2][col]
                updated_green[2][col] = current_red[2][col]
                updated_orange[2][col] = current_green[2][col]
            else:
                updated_orange[2][col] = current_blue[2][col]
                updated_blue[2][col] = current_red[2][col]
                updated_red[2][col] = current_green[2][col]
                updated_green[2][col] = current_orange[2][col]

        # Rotate the yellow face
        current_yellow = self.state["yellow"].copy()
        updated_yellow = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_yellow[row][col] = current_yellow[col][2 - row]
                else:
                    updated_yellow[row][col] = current_yellow[2 - col][row]

        # Save the updated faces back to the cube's state
        self.save_face("blue", updated_blue)
        self.save_face("red", updated_red)
        self.save_face("green", updated_green)
        self.save_face("orange", updated_orange)
        self.save_face("yellow", updated_yellow)

    def rotate_front_face(self, is_prime):
        # Make a copy of current faces that will be affected by the move
        current_red = self.state["red"].copy()
        current_white = self.state["white"].copy()
        current_orange = self.state["orange"].copy()
        current_yellow = self.state["yellow"].copy()
        updated_red = current_red.copy()
        updated_white = current_white.copy()
        updated_orange = current_orange.copy()
        updated_yellow = current_yellow.copy()

        # Iterate to change the cell colors on the edge
        # The right edge of the red face is moved to the yellow face
        # The bottom edge of the white face is moved to the red face
        # The left edge of the orange face is moved to the white face
        # The top edge of the yellow face is moved to the orange face
        for row in range(3):
            row_reversed = 2 - row
            if is_prime:
                updated_yellow[0][row] = current_red[row][2]
                updated_red[row][2] = current_white[2][row_reversed]
                updated_white[2][row] = current_orange[row][0]
                updated_orange[row][0] = current_yellow[0][row_reversed]
            else:
                updated_red[row][2] = current_yellow[0][row]
                updated_white[2][row] = current_red[row_reversed][2]
                updated_orange[row][0] = current_white[2][row]
                updated_yellow[0][row] = current_orange[row_reversed][0]

        # Rotate the blue face
        current_blue = self.state["blue"].copy()
        updated_blue = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_blue[row][col] = current_blue[col][2 - row]
                else:
                    updated_blue[row][col] = current_blue[2 - col][row]

        # Save the updated faces back to the cube's state
        self.save_face("red", updated_red)
        self.save_face("white", updated_white)
        self.save_face("orange", updated_orange)
        self.save_face("yellow", updated_yellow)
        self.save_face("blue", updated_blue)

    def rotate_back_face(self, is_prime):
        # Make a copy of current faces that will be affected by the move
        current_red = self.state["red"].copy()
        current_white = self.state["white"].copy()
        current_orange = self.state["orange"].copy()
        current_yellow = self.state["yellow"].copy()
        updated_red = current_red.copy()
        updated_white = current_white.copy()
        updated_orange = current_orange.copy()
        updated_yellow = current_yellow.copy()

        # Iterate to change the cell colors on the edge
        # The left edge of the red face is moved to the yellow face
        # The top edge of the white face is moved to the red face
        # The right edge of the orange face is moved to the white face
        # The bottom edge of the yellow face is moved to the orange face
        for row in range(3):
            row_reversed = 2 - row
            if is_prime:
                updated_red[row][0] = current_yellow[2][row]
                updated_white[0][row] = current_red[row_reversed][0]
                updated_orange[row][2] = current_white[0][row]
                updated_yellow[2][row] = current_orange[row_reversed][2]
            else:
                updated_yellow[2][row] = current_red[row][0]
                updated_red[row][0] = current_white[0][row_reversed]
                updated_white[0][row] = current_orange[row][2]
                updated_orange[row][2] = current_yellow[2][row_reversed]

        # Rotate the green face
        current_green = self.state["green"].copy()
        updated_green = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_green[row][col] = current_green[col][2 - row]
                else:
                    updated_green[row][col] = current_green[2 - col][row]

        # Save the updated faces back to the cube's state
        self.save_face("red", updated_red)
        self.save_face("white", updated_white)
        self.save_face("orange", updated_orange)
        self.save_face("yellow", updated_yellow)
        self.save_face("green", updated_green)

        return

    def copy(self, cube_to_copy):
        for color in self.state:
            self.state[color] = cube_to_copy.get_face(color).copy()
        return


cube = Cube()


# Returns the boolean value of contour_a being inside contour_b.
def contour_in_contour(contour_a, contour_b):
    # Iterate over all points in contour_a
    for point_as_array in contour_a:
        # Convert the point to a tuple of floats
        point = (float(point_as_array[0][0]), float(point_as_array[0][1]))
        if cv2.pointPolygonTest(contour_b, point, False) < 0:
            return False
    # If all points of contour_a are inside contour_b, return True
    return True


def detection_completed(face):
    # Return True if none of the elements in the face are "init"
    return np.all(face != "init")


def initialize_face():
    return np.full((3, 3), "init", dtype='<U6')


def parents_inside_face(contour_array, hierarchy_array, index, contour_face):
    # Find the hierarchy information for a given index
    hierarchy_i = hierarchy_array[0, index]
    index_of_parent = hierarchy_i[3]
    # If there is no parent, return False
    if index_of_parent == -1:
        return False
    # If the parent is inside contour_face, return True
    contour_parent = contour_array[index_of_parent]
    return contour_in_contour(contour_parent, contour_face)


def get_masks(img):
    # Convert image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = hsv_img.astype(np.uint8)  # Convert to uint8 for NumPy operations
    masks = []
    # Generate a mask for each color range
    for color, (lower, upper) in colour_ranges.items():
        mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
        masks.append(mask)
    return masks


def get_piece_contours_info(masks):
    piece_contours_info = {
        color: cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for color, mask in zip(["blue", "red", "green", "orange", "yellow", "white"], masks)
    }
    return piece_contours_info


# This function finds the contours of each piece of the Rubik's cube
def get_instruction_text(color):
    # Dictionary that maps each cube color to the color of the center facing up
    center_facing_up = {
        "blue": "white",
        "red": "white",
        "green": "white",
        "orange": "white",
        "white": "green",
        "yellow": "blue"
    }
    # Return a string representing the instruction text for the given color
    return f"Show the {color} centered face with {center_facing_up[color]} center facing up"


# This function is used for drawing text over the image
def draw_text(img, text, position):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_x, text_y = position
    background_color = (0, 0, 0)
    text_color = (255, 255, 255)
    text_size, _ = cv2.getTextSize(text, font_face, font_scale, font_thickness)
    text_width, text_height = text_size
    text_position = point_type_converter((text_x, text_y + 1.2 * text_height))
    background_top_left = (text_x, text_y)
    background_bottom_right = point_type_converter((text_x + text_width, text_y + 1.5 * text_height))

    cv2.rectangle(img, background_top_left, background_bottom_right, background_color, -1)
    cv2.putText(img, text, text_position, font_face, font_scale, text_color, font_thickness)
    return


# This function is used for cube detection from video
def detect_cube(video):
    for color in COLORS:
        while True:
            print(cube.state)
            cmd, face = detect_face(video, color)
            if cmd in ["quit", "failed"]:
                return cmd
            elif cmd == "restart":
                continue
            elif cmd == "completed":
                cube.save_face(color, face)
                break
            else:
                raise ValueError(f"Unexpected command: {cmd}")
    return "detected"


# Initializes area values to a relative maximum
def initialize_areas(relative_max=1.0):
    return np.full((3, 3), relative_max)


# Function to check if two faces match
def faces_match(face_a, face_b):
    return np.array_equal(face_a, face_b)


def find_face_and_get_centers(img, margin=20):
    # Convert the image from BGR to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define lower and upper thresholds for color range
    lower_threshold = np.array([0, 0, 150])
    upper_threshold = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_img, lower_threshold, upper_threshold)
    # Find contours in the binary mask
    # cv2.imshow("binary",mask)
    contours, hiearchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Check if any contours are found
    if len(contours) > 0:
        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            # Filter contours based on area range
            if not (10000 < area < 40000):
                continue
            # Fit a rotated rectangle to the contour
            rect = cv2.minAreaRect(contour)
            # Calculate the rotation angle in degrees and radians
            rotation_angle_deg = abs(rect[2])
            rotation_angle_rad = np.deg2rad(rotation_angle_deg)
            center = rect[0]
            # Filter contours based on center distance and rotation angle
            if cv2.norm(center, IMG_CENTER) > 50 or 30 < rotation_angle_deg < 60:
                continue
            # Calculate the bounding box of the rotated rectangle
            bounding_box = cv2.boxPoints(rect)
            bounding_box = np.intp(bounding_box)
            # Check if the contour is rectangle-shaped
            is_rectangle_shaped = True
            for point_as_array in contour:
                point = (float(point_as_array[0][0]), float(point_as_array[0][1]))
                dist = cv2.pointPolygonTest(bounding_box, point, True)
                if dist > margin:
                    is_rectangle_shaped = False
            # Skip non-rectangle-shaped contours
            if not is_rectangle_shaped:
                continue
            # Calculate step sizes  based on rotation angle
            if pi / 3 <= rotation_angle_rad <= pi / 2:
                h, w = rect[1]
                if abs(w - h) > margin:
                    continue
                step_size_x = w // 6
                step_size_y = h // 6
                top_left = bounding_box[0]
                dx_x = sin(rotation_angle_rad) * step_size_x
                dx_y = cos(rotation_angle_rad) * step_size_y
                dy_x = -cos(rotation_angle_rad) * step_size_x
                dy_y = sin(rotation_angle_rad) * step_size_y

            else:
                w, h = rect[1]
                if abs(w - h) > margin:
                    continue
                step_size_x = w // 6
                step_size_y = h // 6
                top_left = bounding_box[1]
                dx_x = cos(rotation_angle_rad) * step_size_x
                dx_y = -sin(rotation_angle_rad) * step_size_y
                dy_x = sin(rotation_angle_rad) * step_size_x
                dy_y = cos(rotation_angle_rad) * step_size_y

            x_0, y_0 = top_left
            # Initialize an empty array to store piece centers
            piece_centers = np.empty((3, 3), dtype="f,f")
            cv2.drawContours(img, [bounding_box], 0, (0, 0, 100), 3)
            # Calculate the centers of each piece
            for i in range(3):
                for j in range(3):
                    piece_center_x = x_0 + (2 * j + 1) * dx_x + (2 * i + 1) * dx_y
                    piece_center_y = y_0 + (2 * j + 1) * dy_x + (2 * i + 1) * dy_y
                    piece_centers[i][j] = (piece_center_x, piece_center_y)
                    piece_center_x = int(piece_center_x)
                    piece_center_y = int(piece_center_y)
                    # Draw a circle at each piece center on the image
                    img = cv2.circle(img, (piece_center_x, piece_center_y),
                                     radius=2, color=(0, 0, 255), thickness=-1)
            # Return the flag indicating a face is found, bounding box, and piece centers
            return True, bounding_box, piece_centers

        return False, None, None


def get_img(video):
    is_ok, img = video.read()
    if not is_ok:
        print("Failed to read the video")
        return False, img
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.flip(img, 1)
    return True, img


def print_mirrored_face(og_face):
    mirrored_face = initialize_face()
    for i in range(3):
        for j in range(3):
            j_reversed = abs(j - 2)
            mirrored_face[i][j] = og_face[i][j_reversed]
    print()
    print("face (mirrored as on the screen):")
    print(mirrored_face)


def detect_face(video, target_center_color):
    is_face_detected = False
    is_face_verified = False
    current_face = initialize_face()
    verified_face = initialize_face()
    piece_relative_areas = initialize_areas()
    instruction_message = get_instruction_text(target_center_color)

    while True:
        success, frame = get_img(video)
        if not success:
            return "failed", None

        draw_text(frame, instruction_message, position=(10, 10))

        if is_face_detected == is_face_verified:
            color_masks = get_masks(frame)
            face_found, face_contour, piece_centers = find_face_and_get_centers(frame)
            piece_contours_info = get_piece_contours_info(color_masks)

            if not face_found:
                current_face = initialize_face()
                piece_relative_areas = initialize_areas()
            else:
                face_area = cv2.contourArea(face_contour)
                min_piece_area = 0.08 * face_area
                piece_contours_info = get_piece_contours_info(color_masks)
                for color, (contours, hierarchy) in piece_contours_info.items():
                    for contour_index, contour in enumerate(contours):
                        contour_area = cv2.contourArea(contour)
                        if len(contour) > 3 and contour_area >= min_piece_area:
                            for row in range(3):
                                for col in range(3):
                                    col_reversed = 2 - col
                                    piece_center = piece_centers[row][col]
                                    if cv2.pointPolygonTest(contour, piece_center, False) == 1:
                                        relative_area = contour_area / face_area
                                        previous_area = piece_relative_areas[row][col_reversed]

                                        if (relative_area <= previous_area and
                                                not parents_inside_face(contours, hierarchy,
                                                                        contour_index, face_contour)):
                                            if row == 1 and col == 1:
                                                if current_face[row][col] != color:
                                                    current_face = initialize_face()
                                                    piece_relative_areas = initialize_areas()
                                                    piece_contours_info = OrderedDict(
                                                        reversed(list(piece_contours_info.items())))
                                                if color == target_center_color or is_face_verified:
                                                    current_face[row][col] = color
                                                    piece_relative_areas[row][col] = relative_area
                                            elif current_face[1][1] == target_center_color or is_face_verified:
                                                current_face[row][col_reversed] = color
                                                piece_relative_areas[row][col_reversed] = relative_area

            if detection_completed(current_face) and not is_face_detected:
                print_mirrored_face(current_face)
                instruction_message = "Correctly detected? y/n?"
                is_face_detected = True

            if is_face_verified:
                if target_center_color == "white":
                    # If the center of the face is white
                    instruction_message = get_instruction_text("yellow")
                    # Get text instruction for turning the cube to the yellow face
                    if detection_completed(current_face):
                        draw_arrows(frame, "x'", piece_centers)
                        # Draw arrows indicating the rotation
                        if current_face[1][1] == "yellow":
                            # If the center of the currently detected face is yellow
                            return "completed", verified_face

                if target_center_color == "yellow":
                    # If the center of the face is yellow
                    instruction_message = get_instruction_text("blue")
                    # Get text instruction for turning the cube to the blue face
                    if detection_completed(current_face):
                        draw_arrows(frame, "x", piece_centers)
                        if current_face[1][1] == "blue":
                            # If the center of the currently detected face is blue
                            return "completed", verified_face

                if target_center_color == "blue":
                    # If the center of the face is blue
                    instruction_message = get_instruction_text("red")
                    # Get text instruction for turning the cube to the red face
                    if detection_completed(current_face):
                        draw_arrows(frame, "y'", piece_centers)
                        if current_face[1][1] == "red":
                            # If the center of the currently detected face is red
                            return "completed", verified_face

                if target_center_color == "red":
                    # If the center of the face is red
                    instruction_message = get_instruction_text("green")
                    # Get text instruction for turning the cube to the green face
                    if detection_completed(current_face):
                        draw_arrows(frame, "y'", piece_centers)
                        if current_face[1][1] == "green":
                            # If the center of the currently detected face is green
                            return "completed", verified_face

                if target_center_color == "green":
                    # If the center of the face is green
                    instruction_message = get_instruction_text("orange")
                    # Get text instruction for turning the cube to the orange face
                    if detection_completed(current_face):
                        draw_arrows(frame, "y'", piece_centers)
                        if current_face[1][1] == "orange":
                            # If the center of the currently detected face is orange
                            return "completed", verified_face

                if target_center_color == "orange":
                    # If the center of the face is orange
                    instruction_message = get_instruction_text("green")
                    # Get text instruction for turning the cube to the green face
                    if detection_completed(current_face):
                        draw_arrows(frame, "y", piece_centers)
                        if current_face[1][1] == "green":
                            # If the center of the currently detected face is green
                            return "completed", verified_face

        if not is_face_verified:
            draw_face(frame, current_face)

        cv2.imshow('CUBE SOLVER', frame)
        pressed_key = cv2.waitKey(1)

        if pressed_key == ord('q'):
            print("the program has been closed")
            return "quit", None
        if pressed_key == ord('y') and is_face_detected:
            verified_face = current_face.copy()
            is_face_verified = True
        if pressed_key == ord('n'):
            return "restart", current_face


def point_type_converter(point, float_to_int=True):
    x = point[0]
    y = point[1]
    if float_to_int:
        point = (int(x), int(y))
    return point


def draw_arrows(img, turn, piece_centers):
    # Convert the piece center coordinates into an integer type
    top_left = point_type_converter(piece_centers[0][0])
    top_mid = point_type_converter(piece_centers[0][1])
    top_right = point_type_converter(piece_centers[0][2])
    mid_left = point_type_converter(piece_centers[1][0])
    mid_right = point_type_converter(piece_centers[1][2])
    bottom_left = point_type_converter(piece_centers[2][0])
    bottom_mid = point_type_converter(piece_centers[2][1])
    bottom_right = point_type_converter(piece_centers[2][2])

    arrow_thickness = 3
    arrow_color = (50, 0, 0)
    # Draw arrows on the image, according to the `turn` argument
    # 'U' means a U face turn - draw an arrow from the top left to the top right
    # 'U'' means a U' face turn - draw an arrow from the top right to the top left
    # Similar logic applies for 'D', 'D'', 'R', 'R'', 'L', 'L'', 'B', 'B'', 'F', 'F'', 'y', 'y'', 'x', 'x''
    # 'y', 'y'', 'x', 'x'' turns mean rotations of the entire cube, arrows are drawn on all the faces
    if turn == "U":
        cv2.arrowedLine(img, top_left, top_right, arrow_color, arrow_thickness)
    if turn == "U'":
        cv2.arrowedLine(img, top_right, top_left, arrow_color, arrow_thickness)
    if turn == "D":
        cv2.arrowedLine(img, bottom_right, bottom_left, arrow_color, arrow_thickness)
    if turn == "D'":
        cv2.arrowedLine(img, bottom_left, bottom_right, arrow_color, arrow_thickness)
    if turn == "R" or turn == "B":
        cv2.arrowedLine(img, top_right, bottom_right, arrow_color, arrow_thickness)
    if turn == "R'" or turn == "B'":
        cv2.arrowedLine(img, bottom_right, top_right, arrow_color, arrow_thickness)
    if turn == "L" or turn == "F":
        cv2.arrowedLine(img, bottom_left, top_left, arrow_color, arrow_thickness)
    if turn == "L'" or turn == "F'":
        cv2.arrowedLine(img, top_left, bottom_left, arrow_color, arrow_thickness)
    if turn == "y":
        cv2.arrowedLine(img, top_left, top_right, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, mid_left, mid_right, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, bottom_left, bottom_right, arrow_color, arrow_thickness)
    if turn == "y'":
        cv2.arrowedLine(img, top_right, top_left, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, mid_right, mid_left, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, bottom_right, bottom_left, arrow_color, arrow_thickness)
    if turn == "x":
        cv2.arrowedLine(img, top_right, bottom_right, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, top_mid, bottom_mid, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, top_left, bottom_left, arrow_color, arrow_thickness)
    if turn == "x'":
        cv2.arrowedLine(img, bottom_right, top_right, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, bottom_mid, top_mid, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, bottom_left, top_left, arrow_color, arrow_thickness)

    return


def draw_face(img, face):
    # Define BGR values for each potential color of a cube piece
    colors_bgr = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "orange": (0, 123, 255),
        "yellow": (0, 255, 255),
        "white": (255, 255, 255),
        "gray": (90, 90, 90),
        "black": (0, 0, 0)
    }
    # Define the upper left point origin of the 3x3 face to be drawn
    origin_face = (500, 40)
    piece_size = 30
    # Draw each cube piece of the face
    for i in range(3):
        for j in range(3):
            j_reversed = 2 - j  # Used to mirror the face correctly (reverse columns)
            origin_piece = (origin_face[0] + j * piece_size, origin_face[1] + i * piece_size)
            end_point_piece = (origin_piece[0] + piece_size, origin_piece[1] + piece_size)
            color_name = face[i][j_reversed] if face[i][j_reversed] != "init" else "gray"
            color_value = colors_bgr[color_name]
            # Calculate the bottom right corner of the current piece
            # Assign gray to uninitialized colors
            cv2.rectangle(img, origin_piece, end_point_piece, color_value, -1)
            # Draw a filled rectangle representing a piece of the cube
    # Draw the borders around each piece
    for i in range(3):
        for j in range(3):
            origin_piece = (origin_face[0] + j * piece_size, origin_face[1] + i * piece_size)
            end_point_piece = (origin_piece[0] + piece_size, origin_piece[1] + piece_size)
            if i < 3 and j < 3:
                cv2.rectangle(img, origin_piece, end_point_piece, colors_bgr["black"], 2)
            else:
                cv2.rectangle(img, origin_piece, end_point_piece, colors_bgr["black"], 1)
    draw_text(img, text="2D Face", position=(505, 132))


def execute_turn(video, turn, previous_center_color, is_last_turn):
    # Initialize a flag to check if the Rubik's cube is solved
    is_solved = False
    # Extract the first character of the turn
    turn_letter = turn[0]
    if turn_letter == 'U' or turn_letter == 'D':
        valid_center_colors = ["green", "red"]
    elif turn_letter == 'R' or turn_letter == 'L':
        valid_center_colors = ["green"]
    else:
        valid_center_colors = ["red"]
    # Determine the current center color based on the previous center color
    if previous_center_color in valid_center_colors:
        current_center_color = previous_center_color
    else:
        current_center_color = valid_center_colors[0]
    # Store the state of the red and green faces before the move
    red_face_before_turn = cube.get_face("red")
    green_face_before_turn = cube.get_face("green")
    # Assign the appropriate face before the turn
    if current_center_color == "red":
        face_before_turn = red_face_before_turn
    else:
        face_before_turn = green_face_before_turn
    # Execute the turn on the cube and get face after the turn
    cube.call_turn(turn)
    face_after_turn = cube.get_face(current_center_color)
    detected_face = initialize_face()
    piece_relative_areas = initialize_areas()
    instruction_message = get_instruction_text(current_center_color)
    # Detects the new face with the same logic as before.
    while True:
        success, frame = get_img(video)
        if not success:
            return "failed", None
        if not is_solved:
            draw_text(frame, instruction_message, position=(10, 10))
            color_masks = get_masks(frame)
            face_detected, face_contour, detected_piece_centers = find_face_and_get_centers(frame)
            if not face_detected:
                detected_face = initialize_face()
                piece_relative_areas = initialize_areas()
            else:
                face_area = cv2.contourArea(face_contour)
                min_piece_area = 0.08 * face_area
                piece_contours_info = get_piece_contours_info(color_masks)
                for color, (contours, hierarchy) in piece_contours_info.items():
                    for contour_index, contour in enumerate(contours):
                        contour_area = cv2.contourArea(contour)
                        if len(contour) > 3 and contour_area >= min_piece_area:
                            for row in range(3):
                                for col in range(3):
                                    col_reversed = 2 - col
                                    piece_center = detected_piece_centers[row][col]
                                    if cv2.pointPolygonTest(contour, piece_center, False) == 1:
                                        relative_area = contour_area / face_area
                                        previous_area = piece_relative_areas[row][col_reversed]
                                        if (relative_area <= previous_area and
                                                not parents_inside_face(contours, hierarchy,
                                                                        contour_index, face_contour)):
                                            if row == 1 and col == 1:
                                                if detected_face[row][col] != color:
                                                    detected_face = initialize_face()
                                                    piece_relative_areas = initialize_areas()
                                                if color in ["red", "green"]:
                                                    detected_face[row][col] = color
                                                    piece_relative_areas[row][col] = relative_area
                                            elif detected_face[1][1] in ["red", "green"]:
                                                detected_face[row][col_reversed] = color
                                                piece_relative_areas[row][col_reversed] = relative_area
                if detection_completed(detected_face):
                    if faces_match(detected_face, face_before_turn):
                        draw_arrows(frame, turn, detected_piece_centers)
                    elif faces_match(detected_face, face_after_turn):
                        if not is_last_turn:
                            return "completed", current_center_color
                        else:
                            is_solved = True
                    elif (faces_match(detected_face, green_face_before_turn) and
                          current_center_color == "red"):
                        draw_arrows(frame, "y", detected_piece_centers)
                    elif (faces_match(detected_face, red_face_before_turn) and
                          current_center_color == "green"):
                        draw_arrows(frame, "y'", detected_piece_centers)
                        # If the cube is solved, display a completion message
                        # and wait for the 'q' key to be pressed to finish
                        # or continue capturing frames from the video feed and processing
        else:
            draw_text(frame, "THE CUBE IS SOLVED!", position=(250, 10))
            draw_text(frame, "Press 'q' to close the window.", position=(170, 35))
        cv2.imshow('CUBE SOLVER', frame)
        pressed_key = cv2.waitKey(1)
        if pressed_key == ord('q'):
            return "quit", None


# Function to reformat the solution string. If a turn operation is denoted with '2' at the end (meaning it has to be
# performed twice), it duplicates that operation in the list.
def reform_solution(solution_str):
    return [turn[0] if len(turn) > 1 and turn[1] == "2" else turn for turn in solution_str.split() for _ in
            range(2 if len(turn) > 1 and turn[1] == "2" else 1)]


# This function solves the Rubik's cube by executing each turn operation in the solution list
def solve_cube(video):
    solution = reform_solution(cube.get_solution())
    previous_center_color = "none"
    for i, turn in enumerate(solution):
        last_turn = (i == len(solution) - 1)
        cmd, previous_center_color = execute_turn(video, turn, previous_center_color, last_turn)

        if cmd in ["quit", "failed"]:
            return cmd
    return "solved"


def main():
    # Opens the default camera
    video = cv2.VideoCapture(0)
    try:
        cmd = detect_cube(video)
        # If the cube is detected successfully, proceed to solve the cube
        if cmd == "detected":
            cmd = solve_cube(video)
        # If the cmd is "quit", n 'Program finished due to keyboard command'
        if cmd == "quit":
            print("Program finished due to keyboard command")
        # If the cmd is "failed", print 'Program finished due to unexpected error
        elif cmd == "failed":
            print("Program finished due to unexpected error")
        else:
            print("Cube solved successfully")
    finally:
        video.release()
        cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    main()
