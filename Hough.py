""" Rojeh edit """

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv
from TestsResults import *


def hough_lines_horizontal(im, lower_threshold, highr_threshold):
    im_l = im

    edge_detected = cv2.Canny(im_l, lower_threshold, highr_threshold, apertureSize=3)
    lines = cv2.HoughLines(edge_detected, 1, np.pi / 180, 80, min_theta=np.pi / 2 - np.pi / 60, max_theta=np.pi / 2 + np.pi / 60)

    lines_list = []

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

            lines_list.append([x1, y1, x2, y2])

    lines = []
    for line in lines_list:
        x1, y1, x2, y2 = line
        mid_y = int((y1 + y2) / 2)
        res = any(abs((x[1] + x[3])/2 - mid_y) <= 8 for x in lines)
        if not res:
            lines.append(line)

    startEndLines = []
    if len(lines) != 0:
        sorted_lines = sorted(lines, key=lambda x: x[3])
        startEndLines.append(sorted_lines[0])
        cv2.line(im_l, (sorted_lines[0][0], sorted_lines[0][1]), (sorted_lines[0][2], sorted_lines[0][3]),
                 (255, 0, 0), 4)
        if len(sorted_lines) >= 3:
            startEndLines.append(sorted_lines[2])
            cv2.line(im_l, (sorted_lines[2][0], sorted_lines[2][1]), (sorted_lines[2][2], sorted_lines[2][3]),
                    (255, 0, 0), 4)
        return startEndLines

    return None


def transformedImage(frame, coordinates):
    n1, m1 = int(max(coordinates[2][1], coordinates[3][1]) - min(coordinates[0][1], coordinates[1][1]))\
        , int(max(coordinates[1][0], coordinates[2][0]) - min(coordinates[0][0], coordinates[3][0]))
    n1 = max(n1, m1)
    m1 = n1
    dest = np.float32([[0, 0],
                        [n1, 0],
                        [n1, m1],
                        [0, m1]])

    T = cv2.getPerspectiveTransform(coordinates, dest)
    transformed_im = cv2.warpPerspective(frame, T, (m1, n1), flags = cv2.INTER_CUBIC)

    return transformed_im, T


def start(frame, coordinates, toReturn):
    im2, T = transformedImage(frame, coordinates)

    lower_threshold, higher_threshold = 110, 150
    result = hough_lines_horizontal(im2, lower_threshold, higher_threshold)

    if result is None:
        lower_threshold -= 110
        higher_threshold -= 110
        result = hough_lines_horizontal(im2, lower_threshold, higher_threshold)
    if result is None:
        return None

    startPts1 = inv(T) @ [result[0][0], result[0][1], 1]
    startPts2 = inv(T) @ [result[0][2], result[0][3], 1]
    startPts1 /= startPts1[2]
    startPts2 /= startPts2[2]
    startPts = []
    startPts.extend(startPts1[:2])
    startPts.extend(startPts2[:2])
    toReturn.append(startPts)
    if len(result) > 1 and len(toReturn) < 2:
        endPts1 = inv(T) @ [result[1][0], result[1][1], 1]
        endPts2 = inv(T) @ [result[1][2], result[1][3], 1]
        endPts1 /= endPts1[2]
        endPts2 /= endPts2[2]
        endPts = []
        endPts.extend(endPts1[:2])
        endPts.extend(endPts2[:2])
        toReturn.append(endPts)

    return toReturn


def configureCoords(path, frame, coords, kerem=False):
    toReturn = []

    saved_value = save_evaluation(path, None, 'Start Line', kerem=kerem)
    if saved_value is not None:
        toReturn.append(saved_value)
    saved_value = save_evaluation(path, None, 'End Line', kerem=kerem)
    if saved_value is not None:
        toReturn.append(saved_value)
        return toReturn

    coordinates = coords.copy()
    coordinates[1][0] += 100
    coordinates[0][0] -= 120
    coordinates[1][1] += 30
    coordinates[0][1] += 30
    coordinates.append([int((frame.shape[1]+200) / 2) + 200, frame.shape[0] - 1])
    coordinates.append([int((frame.shape[1] + 200) / 2) - 200, frame.shape[0] - 1])

    coordinates = np.float32(coordinates)

    return start(frame, coordinates, toReturn)