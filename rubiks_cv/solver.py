from collections import OrderedDict

import cv2

from .config import COLORS
from .cube import cube
from .ui import draw_arrows, draw_face, draw_text, get_instruction_text, print_mirrored_face
from .utils import (
    detection_completed,
    faces_match,
    initialize_areas,
    initialize_face,
)
from .vision import (
    find_face_and_get_centers,
    get_img,
    get_masks,
    get_piece_contours_info,
    parents_inside_face,
)


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
                                        if (
                                            relative_area <= previous_area
                                            and not parents_inside_face(
                                                contours, hierarchy, contour_index, face_contour
                                            )
                                        ):
                                            if row == 1 and col == 1:
                                                if current_face[row][col] != color:
                                                    current_face = initialize_face()
                                                    piece_relative_areas = initialize_areas()
                                                    piece_contours_info = OrderedDict(
                                                        reversed(list(piece_contours_info.items()))
                                                    )
                                                if color == target_center_color or is_face_verified:
                                                    current_face[row][col] = color
                                                    piece_relative_areas[row][col] = relative_area
                                            elif (
                                                current_face[1][1] == target_center_color
                                                or is_face_verified
                                            ):
                                                current_face[row][col_reversed] = color
                                                piece_relative_areas[row][
                                                    col_reversed
                                                ] = relative_area

            if detection_completed(current_face) and not is_face_detected:
                print_mirrored_face(current_face)
                instruction_message = "Correctly detected? y/n?"
                is_face_detected = True

            if is_face_verified:
                if target_center_color == "white":
                    instruction_message = get_instruction_text("yellow")
                    if detection_completed(current_face):
                        draw_arrows(frame, "x'", piece_centers)
                        if current_face[1][1] == "yellow":
                            return "completed", verified_face

                if target_center_color == "yellow":
                    instruction_message = get_instruction_text("blue")
                    if detection_completed(current_face):
                        draw_arrows(frame, "x", piece_centers)
                        if current_face[1][1] == "blue":
                            return "completed", verified_face

                if target_center_color == "blue":
                    instruction_message = get_instruction_text("red")
                    if detection_completed(current_face):
                        draw_arrows(frame, "y'", piece_centers)
                        if current_face[1][1] == "red":
                            return "completed", verified_face

                if target_center_color == "red":
                    instruction_message = get_instruction_text("green")
                    if detection_completed(current_face):
                        draw_arrows(frame, "y'", piece_centers)
                        if current_face[1][1] == "green":
                            return "completed", verified_face

                if target_center_color == "green":
                    instruction_message = get_instruction_text("orange")
                    if detection_completed(current_face):
                        draw_arrows(frame, "y'", piece_centers)
                        if current_face[1][1] == "orange":
                            return "completed", verified_face

                if target_center_color == "orange":
                    instruction_message = get_instruction_text("green")
                    if detection_completed(current_face):
                        draw_arrows(frame, "y", piece_centers)
                        if current_face[1][1] == "green":
                            return "completed", verified_face

        if not is_face_verified:
            draw_face(frame, current_face)

        cv2.imshow("CUBE SOLVER", frame)
        pressed_key = cv2.waitKey(1)

        if pressed_key == ord("q"):
            print("the program has been closed")
            return "quit", None
        if pressed_key == ord("y") and is_face_detected:
            verified_face = current_face.copy()
            is_face_verified = True
        if pressed_key == ord("n"):
            return "restart", current_face


def execute_turn(video, turn, previous_center_color, is_last_turn):
    is_solved = False
    turn_letter = turn[0]
    if turn_letter == "U" or turn_letter == "D":
        valid_center_colors = ["green", "red"]
    elif turn_letter == "R" or turn_letter == "L":
        valid_center_colors = ["green"]
    else:
        valid_center_colors = ["red"]
    if previous_center_color in valid_center_colors:
        current_center_color = previous_center_color
    else:
        current_center_color = valid_center_colors[0]
    red_face_before_turn = cube.get_face("red")
    green_face_before_turn = cube.get_face("green")
    if current_center_color == "red":
        face_before_turn = red_face_before_turn
    else:
        face_before_turn = green_face_before_turn
    cube.call_turn(turn)
    face_after_turn = cube.get_face(current_center_color)
    detected_face = initialize_face()
    piece_relative_areas = initialize_areas()
    instruction_message = get_instruction_text(current_center_color)
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
                                        if (
                                            relative_area <= previous_area
                                            and not parents_inside_face(
                                                contours, hierarchy, contour_index, face_contour
                                            )
                                        ):
                                            if row == 1 and col == 1:
                                                if detected_face[row][col] != color:
                                                    detected_face = initialize_face()
                                                    piece_relative_areas = initialize_areas()
                                                if color in ["red", "green"]:
                                                    detected_face[row][col] = color
                                                    piece_relative_areas[row][col] = relative_area
                                            elif detected_face[1][1] in ["red", "green"]:
                                                detected_face[row][col_reversed] = color
                                                piece_relative_areas[row][
                                                    col_reversed
                                                ] = relative_area
                if detection_completed(detected_face):
                    if faces_match(detected_face, face_before_turn):
                        draw_arrows(frame, turn, detected_piece_centers)
                    elif faces_match(detected_face, face_after_turn):
                        if not is_last_turn:
                            return "completed", current_center_color
                        else:
                            is_solved = True
                    elif (
                        faces_match(detected_face, green_face_before_turn)
                        and current_center_color == "red"
                    ):
                        draw_arrows(frame, "y", detected_piece_centers)
                    elif (
                        faces_match(detected_face, red_face_before_turn)
                        and current_center_color == "green"
                    ):
                        draw_arrows(frame, "y'", detected_piece_centers)
        else:
            draw_text(frame, "THE CUBE IS SOLVED!", position=(250, 10))
            draw_text(frame, "Press 'q' to close the window.", position=(170, 35))
        cv2.imshow("CUBE SOLVER", frame)
        pressed_key = cv2.waitKey(1)
        if pressed_key == ord("q"):
            return "quit", None


def reform_solution(solution_str):
    return [
        turn[0] if len(turn) > 1 and turn[1] == "2" else turn
        for turn in solution_str.split()
        for _ in range(2 if len(turn) > 1 and turn[1] == "2" else 1)
    ]


def solve_cube(video):
    solution = reform_solution(cube.get_solution())
    previous_center_color = "none"
    for i, turn in enumerate(solution):
        last_turn = i == len(solution) - 1
        cmd, previous_center_color = execute_turn(video, turn, previous_center_color, last_turn)
        if cmd in ["quit", "failed"]:
            return cmd
    return "solved"
