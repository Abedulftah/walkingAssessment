import cv2
import numpy as np
from scipy.linalg import inv


def hough_lines_horizontal(im, lower_threshold, highr_threshold):
    im_l = im

    edge_detected = cv2.Canny(im_l, lower_threshold, highr_threshold, apertureSize=3)
    lines = cv2.HoughLines(edge_detected, 1, np.pi / 180, 80, min_theta=np.pi / 2 - np.pi / 60, max_theta=np.pi / 2 + np.pi / 60)

    minCol = float('inf')
    begX, begY, endX, endY = None, None, None, None

    if lines is not None and lines.any():
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 - 1000 * b)
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 + 1000 * b)
            y2 = int(y0 - 1000 * a)

            # Check if endpoints are on screen, and adjust them if necessary
            h, w = im_l.shape[:2]
            if x1 < 0:
                y1 = int((0 - x1) * (y2 - y1) / (x2 - x1) + y1)
                x1 = 0
            elif x1 >= w:
                y1 = int((w - 1 - x1) * (y2 - y1) / (x2 - x1) + y1)
                x1 = w - 1
            if y1 < 0:
                x1 = int((0 - y1) * (x2 - x1) / (y2 - y1) + x1)
                y1 = 0
            elif y1 >= h:
                x1 = int((h - 1 - y1) * (x2 - x1) / (y2 - y1) + x1)
                y1 = h - 1
            if x2 < 0:
                y2 = int((0 - x2) * (y1 - y2) / (x1 - x2) + y2)
                x2 = 0
            elif x2 >= w:
                y2 = int((w - 1 - x2) * (y1 - y2) / (x1 - x2) + y2)
                x2 = w - 1
            if y2 < 0:
                x2 = int((0 - y2) * (x1 - x2) / (y1 - y2) + x2)
                y2 = 0
            elif y2 >= h:
                x2 = int((h - 1 - y2) * (x1 - x2) / (y1 - y2) + x2)
                y2 = h - 1

            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)

            k_tuple = tuple(im_l[mid_y][mid_x])
            if not k_tuple in color_dict or color_dict[k_tuple] <= minCol:
                minCol = color_dict[k_tuple] if k_tuple in color_dict else 0
                begX, begY = x1, y1
                endX, endY = x2, y2

    if begX != None:
        return [begX, begY, endX, endY]

    return None


def transformedImage(frame, coordinates):
    n, m = frame.shape[:2]
    n1, m1 = int(max(coordinates[2][1], coordinates[3][1]) - min(coordinates[0][1], coordinates[1][1]))\
        , int(max(coordinates[1][0], coordinates[2][0]) - min(coordinates[0][0], coordinates[3][0]))
    n1 = max(n1,m1)
    m1=n1
    dest = np.float32([[0, 0],
                        [n1, 0],
                        [n1, m1],
                        [0, m1]])
    T = cv2.getPerspectiveTransform(coordinates, dest)
    transformed_im = cv2.warpPerspective(frame, T, (m1, n1), flags = cv2.INTER_CUBIC)

    return transformed_im, T

def hashColors(frame):
    for m in frame:
        for k in m:
            k_tuple = tuple(k)
            if k_tuple in color_dict:
                color_dict[k_tuple] += 1
            else:
                color_dict[k_tuple] = 0


def start(frame, coordinates):
    global color_dict
    color_dict = {}

    n, m = frame.shape[:2]

    im2, T = transformedImage(frame, coordinates)
    hashColors(im2)

    result = []
    lower_threshold, higher_threshold = 180, 240
    result = hough_lines_horizontal(im2, lower_threshold, higher_threshold)

    if result == None:
        lower_threshold -= 110
        higher_threshold -= 110
        result = hough_lines_horizontal(im2, lower_threshold, higher_threshold)
    if result == None:
        return None

    startPt = inv(T) @ [result[0], result[1], 1]
    endPt = inv(T) @ [result[2], result[3], 1]
    startPt /= startPt[2]
    endPt /= endPt[2]

    toReturn = []
    toReturn.append([int(startPt[0]), int(endPt[0])])
    toReturn.append([int(startPt[1]), int(endPt[1])])

    return toReturn


def configureCoords(frame, coordinates):
    coordinates[1][0] += 100
    coordinates[0][0] -= 120
    coordinates[1][1] += 30
    coordinates[0][1] += 30
    coordinates.append([int((frame.shape[1]+200) / 2) +200, frame.shape[0] - 1])
    coordinates.append([int((frame.shape[1] + 200) / 2) - 200, frame.shape[0] - 1])

    coordinates = np.float32(coordinates)

    return start(frame, coordinates)