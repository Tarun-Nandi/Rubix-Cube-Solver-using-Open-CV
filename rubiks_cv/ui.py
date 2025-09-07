import cv2

from .utils import point_type_converter


def get_instruction_text(color):
    center_facing_up = {
        "blue": "white",
        "red": "white",
        "green": "white",
        "orange": "white",
        "white": "green",
        "yellow": "blue",
    }
    return f"Show the {color} centered face with {center_facing_up[color]} center facing up"


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
    background_bottom_right = point_type_converter(
        (text_x + text_width, text_y + 1.5 * text_height)
    )
    cv2.rectangle(img, background_top_left, background_bottom_right, background_color, -1)
    cv2.putText(img, text, text_position, font_face, font_scale, text_color, font_thickness)
    return


def draw_arrows(img, turn, piece_centers):
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
    colors_bgr = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "orange": (0, 123, 255),
        "yellow": (0, 255, 255),
        "white": (255, 255, 255),
        "gray": (90, 90, 90),
        "black": (0, 0, 0),
    }
    origin_face = (500, 40)
    piece_size = 30
    for i in range(3):
        for j in range(3):
            j_reversed = 2 - j
            origin_piece = (origin_face[0] + j * piece_size, origin_face[1] + i * piece_size)
            end_point_piece = (origin_piece[0] + piece_size, origin_piece[1] + piece_size)
            color_name = face[i][j_reversed] if face[i][j_reversed] != "init" else "gray"
            color_value = colors_bgr[color_name]
            cv2.rectangle(img, origin_piece, end_point_piece, color_value, -1)
    for i in range(3):
        for j in range(3):
            origin_piece = (origin_face[0] + j * piece_size, origin_face[1] + i * piece_size)
            end_point_piece = (origin_piece[0] + piece_size, origin_piece[1] + piece_size)
            if i < 3 and j < 3:
                cv2.rectangle(img, origin_piece, end_point_piece, colors_bgr["black"], 2)
            else:
                cv2.rectangle(img, origin_piece, end_point_piece, colors_bgr["black"], 1)
    draw_text(img, text="2D Face", position=(505, 132))


def print_mirrored_face(og_face):
    from .utils import initialize_face

    mirrored_face = initialize_face()
    for i in range(3):
        for j in range(3):
            j_reversed = abs(j - 2)
            mirrored_face[i][j] = og_face[i][j_reversed]
    print()
    print("face (mirrored as on the screen):")
    print(mirrored_face)
