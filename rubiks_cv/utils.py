import cv2
import numpy as np


def contour_in_contour(contour_a, contour_b):
    for point_as_array in contour_a:
        point = (float(point_as_array[0][0]), float(point_as_array[0][1]))
        if cv2.pointPolygonTest(contour_b, point, False) < 0:
            return False
    return True


def detection_completed(face):
    return np.all(face != "init")


def initialize_face():
    return np.full((3, 3), "init", dtype="<U6")


def initialize_areas(relative_max=1.0):
    return np.full((3, 3), relative_max)


def faces_match(face_a, face_b):
    return np.array_equal(face_a, face_b)


def point_type_converter(point, float_to_int=True):
    x, y = point
    if float_to_int:
        point = (int(x), int(y))
    return point
