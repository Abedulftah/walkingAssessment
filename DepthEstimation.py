import numpy as np


height_avg = 1.60


def get_depth_ratio(frame1, frame2, keypoints1, keypoints2):
    if keypoints1.shape == (0,) or keypoints2.shape == (0,):
        return None

    head_coords1 = frame1.shape[0] * (keypoints1[0][0][:2])
    between_foots_coords1 = frame1.shape[0] * (keypoints1[0][15][:2] +
                                               keypoints1[0][16][:2]) / 2

    head_coords2 = frame2.shape[0] * (keypoints2[0][0][:2])
    between_foots_coords2 = frame2.shape[0] * (keypoints2[0][15][:2] +
                                               keypoints2[0][16][:2]) / 2

    height_pix1 = np.sqrt(
        (head_coords1[0] - between_foots_coords1[0]) ** 2 + (head_coords1[1] - between_foots_coords1[1]) ** 2)
    height_pix2 = np.sqrt(
        (head_coords2[0] - between_foots_coords2[0]) ** 2 + (head_coords2[1] - between_foots_coords2[1]) ** 2)

    depth_ratio = height_pix2 / height_pix1

    if depth_ratio < 0.9:
        return "Getting Further"
    elif depth_ratio > 1.1:
        return "Getting Closer"

    return "Not Moving"


def coord_to_line_distance(coordinate, lineCoords):
    A = (lineCoords[0][0] - lineCoords[1][0]) / (lineCoords[0][1] - lineCoords[1][1])
    B = -1
    C = lineCoords[0][0] - A * lineCoords[0][1]

    return A * coordinate[0] + B * coordinate[1] + C / np.sqrt(A**2 + B**2)
