import kociemba
import numpy as np


def color_to_letter(color):
    color_letter_map = {
        "blue": "F",
        "green": "B",
        "white": "U",
        "yellow": "D",
        "red": "L",
        "orange": "R",
    }
    if color in color_letter_map:
        return color_letter_map[color]
    else:
        raise ValueError(f"Invalid color: {color}")


def parse_turn(turn_str):
    is_prime = False
    if len(turn_str) > 2 or len(turn_str) == 0:
        raise ValueError(f"Invalid turn: {turn_str}")
    if len(turn_str) == 2:
        if turn_str[1] == "'":
            is_prime = True
            turn = turn_str[0]
        elif turn_str[1] == "2":
            turn = turn_str[0] * 2
        else:
            raise ValueError(f"Invalid turn: {turn_str}")
    else:
        turn = turn_str
    return turn, is_prime


def initialize_face():
    return np.full((3, 3), "init", dtype="<U6")


class Cube:
    def __init__(self):
        empty_face = np.empty([3, 3], "<U6")
        self.state = {
            "white": empty_face,
            "orange": empty_face,
            "blue": empty_face,
            "yellow": empty_face,
            "red": empty_face,
            "green": empty_face,
        }

    def save_face(self, color, face):
        self.state[color] = face

    def get_face(self, color):
        return self.state[color]

    def get_solution(self):
        state_str = ""
        for color in self.state:
            face = self.state[color]
            for i in range(3):
                for j in range(3):
                    state_str += color_to_letter(face[i][j])
        solution = kociemba.solve(state_str)
        return solution

    def call_turn(self, turn_str):
        turn, is_prime = parse_turn(turn_str)
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

    # === Rotation methods unchanged ===
    def rotate_right_face(self, is_prime):
        current_blue = self.state["blue"].copy()
        current_white = self.state["white"].copy()
        current_green = self.state["green"].copy()
        current_yellow = self.state["yellow"].copy()
        updated_blue = current_blue.copy()
        updated_white = current_white.copy()
        updated_green = current_green.copy()
        updated_yellow = current_yellow.copy()
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
        current_orange = self.state["orange"].copy()
        updated_orange = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_orange[row][col] = current_orange[col][2 - row]
                else:
                    updated_orange[row][col] = current_orange[2 - col][row]
        self.save_face("blue", updated_blue)
        self.save_face("white", updated_white)
        self.save_face("green", updated_green)
        self.save_face("yellow", updated_yellow)
        self.save_face("orange", updated_orange)

    def rotate_left_face(self, is_prime):
        current_blue = self.state["blue"].copy()
        current_white = self.state["white"].copy()
        current_green = self.state["green"].copy()
        current_yellow = self.state["yellow"].copy()
        updated_blue = current_blue.copy()
        updated_white = current_white.copy()
        updated_green = current_green.copy()
        updated_yellow = current_yellow.copy()
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
        current_red = self.state["red"].copy()
        updated_red = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_red[row][col] = current_red[col][2 - row]
                else:
                    updated_red[row][col] = current_red[2 - col][row]
        self.save_face("blue", updated_blue)
        self.save_face("white", updated_white)
        self.save_face("green", updated_green)
        self.save_face("yellow", updated_yellow)
        self.save_face("red", updated_red)

    def rotate_up_face(self, is_prime):
        current_blue = self.state["blue"].copy()
        current_red = self.state["red"].copy()
        current_green = self.state["green"].copy()
        current_orange = self.state["orange"].copy()
        updated_blue = current_blue.copy()
        updated_red = current_red.copy()
        updated_green = current_green.copy()
        updated_orange = current_orange.copy()
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
        current_white = self.state["white"].copy()
        updated_white = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_white[row][col] = current_white[col][2 - row]
                else:
                    updated_white[row][col] = current_white[2 - col][row]
        self.save_face("blue", updated_blue)
        self.save_face("red", updated_red)
        self.save_face("green", updated_green)
        self.save_face("orange", updated_orange)
        self.save_face("white", updated_white)

    def rotate_down_face(self, is_prime):
        current_blue = self.state["blue"].copy()
        current_red = self.state["red"].copy()
        current_green = self.state["green"].copy()
        current_orange = self.state["orange"].copy()
        updated_blue = current_blue.copy()
        updated_red = current_red.copy()
        updated_green = current_green.copy()
        updated_orange = current_orange.copy()
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
        current_yellow = self.state["yellow"].copy()
        updated_yellow = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_yellow[row][col] = current_yellow[col][2 - row]
                else:
                    updated_yellow[row][col] = current_yellow[2 - col][row]
        self.save_face("blue", updated_blue)
        self.save_face("red", updated_red)
        self.save_face("green", updated_green)
        self.save_face("orange", updated_orange)
        self.save_face("yellow", updated_yellow)

    def rotate_front_face(self, is_prime):
        current_red = self.state["red"].copy()
        current_white = self.state["white"].copy()
        current_orange = self.state["orange"].copy()
        current_yellow = self.state["yellow"].copy()
        updated_red = current_red.copy()
        updated_white = current_white.copy()
        updated_orange = current_orange.copy()
        updated_yellow = current_yellow.copy()
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
        current_blue = self.state["blue"].copy()
        updated_blue = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_blue[row][col] = current_blue[col][2 - row]
                else:
                    updated_blue[row][col] = current_blue[2 - col][row]
        self.save_face("red", updated_red)
        self.save_face("white", updated_white)
        self.save_face("orange", updated_orange)
        self.save_face("yellow", updated_yellow)
        self.save_face("blue", updated_blue)

    def rotate_back_face(self, is_prime):
        current_red = self.state["red"].copy()
        current_white = self.state["white"].copy()
        current_orange = self.state["orange"].copy()
        current_yellow = self.state["yellow"].copy()
        updated_red = current_red.copy()
        updated_white = current_white.copy()
        updated_orange = current_orange.copy()
        updated_yellow = current_yellow.copy()
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
        current_green = self.state["green"].copy()
        updated_green = initialize_face()
        for row in range(3):
            for col in range(3):
                if is_prime:
                    updated_green[row][col] = current_green[col][2 - row]
                else:
                    updated_green[row][col] = current_green[2 - col][row]
        self.save_face("red", updated_red)
        self.save_face("white", updated_white)
        self.save_face("orange", updated_orange)
        self.save_face("yellow", updated_yellow)
        self.save_face("green", updated_green)

    def copy(self, cube_to_copy):
        for color in self.state:
            self.state[color] = cube_to_copy.get_face(color).copy()
        return


# Global cube instance preserved
cube = Cube()
