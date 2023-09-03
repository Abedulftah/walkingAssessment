""" DepthEstimation class helps estimate foot distance from the given line, which helps decide whether the patient
 passed the line or not!"""

import numpy as np

def coord_to_line_distance(coordinate, lineCoords):
    A = (lineCoords[1] - lineCoords[3]) / (lineCoords[0] - lineCoords[2])
    B = -1
    C = lineCoords[1] - A * lineCoords[0]
    return (A * coordinate[0] + B * coordinate[1] + C) / np.sqrt(A**2 + B**2)
