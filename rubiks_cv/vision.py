import cv2
import numpy as np

from .config import IMG_CENTER, IMG_HEIGHT, IMG_WIDTH, colour_ranges


def parents_inside_face(contour_array, hierarchy_array, index, contour_face):
    hierarchy_i = hierarchy_array[0, index]
    index_of_parent = hierarchy_i[3]
    if index_of_parent == -1:
        return False
    contour_parent = contour_array[index_of_parent]
    return (
        cv2.pointPolygonTest(
            contour_face, (float(contour_parent[0][0][0]), float(contour_parent[0][0][1])), False
        )
        >= 0
    )  # conservative


def get_masks(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = hsv_img.astype(np.uint8)
    masks = []
    for _color, (lower, upper) in colour_ranges.items():
        mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
        masks.append(mask)
    return masks


def get_piece_contours_info(masks):
    piece_contours_info = {
        color: cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for color, mask in zip(["blue", "red", "green", "orange", "yellow", "white"], masks)
    }
    return piece_contours_info


def find_face_and_get_centers(img, margin=20):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_threshold = np.array([0, 0, 150])
    upper_threshold = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_img, lower_threshold, upper_threshold)
    contours, hiearchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (10000 < area < 40000):
                continue
            rect = cv2.minAreaRect(contour)
            rotation_angle_deg = abs(rect[2])
            rotation_angle_rad = np.deg2rad(rotation_angle_deg)
            center = rect[0]
            if cv2.norm(center, IMG_CENTER) > 50 or 30 < rotation_angle_deg < 60:
                continue
            bounding_box = cv2.boxPoints(rect)
            bounding_box = np.intp(bounding_box)
            is_rectangle_shaped = True
            for point_as_array in contour:
                point = (float(point_as_array[0][0]), float(point_as_array[0][1]))
                dist = cv2.pointPolygonTest(bounding_box, point, True)
                if dist > margin:
                    is_rectangle_shaped = False
            if not is_rectangle_shaped:
                continue
            if np.pi / 3 <= rotation_angle_rad <= np.pi / 2:
                h, w = rect[1]
                if abs(w - h) > margin:
                    continue
                step_size_x = w // 6
                step_size_y = h // 6
                top_left = bounding_box[0]
                dx_x = np.sin(rotation_angle_rad) * step_size_x
                dx_y = np.cos(rotation_angle_rad) * step_size_y
                dy_x = -np.cos(rotation_angle_rad) * step_size_x
                dy_y = np.sin(rotation_angle_rad) * step_size_y
            else:
                w, h = rect[1]
                if abs(w - h) > margin:
                    continue
                step_size_x = w // 6
                step_size_y = h // 6
                top_left = bounding_box[1]
                dx_x = np.cos(rotation_angle_rad) * step_size_x
                dx_y = -np.sin(rotation_angle_rad) * step_size_y
                dy_x = np.sin(rotation_angle_rad) * step_size_x
                dy_y = np.cos(rotation_angle_rad) * step_size_y
            x_0, y_0 = top_left
            piece_centers = np.empty((3, 3), dtype="f,f")
            cv2.drawContours(img, [bounding_box], 0, (0, 0, 100), 3)
            for i in range(3):
                for j in range(3):
                    piece_center_x = x_0 + (2 * j + 1) * dx_x + (2 * i + 1) * dx_y
                    piece_center_y = y_0 + (2 * j + 1) * dy_x + (2 * i + 1) * dy_y
                    piece_centers[i][j] = (piece_center_x, piece_center_y)
                    piece_center_x = int(piece_center_x)
                    piece_center_y = int(piece_center_y)
                    img = cv2.circle(
                        img,
                        (piece_center_x, piece_center_y),
                        radius=2,
                        color=(0, 0, 255),
                        thickness=-1,
                    )
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
