import tensorflow as tf
import tensorflow_hub as hub
import cv2
import time
import numpy as np
from Hough import *


# I need to check how to use it
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

# a dictionary to connect the coordinates together
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.destroyAllWindows()
        multiPose([y, x])


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 3, (0, 255, 0), -1)


def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


def detect_person(keypoints_with_scores, select):
    global doChange
    y, x, _ = frame.shape
    right_person = None
    min_person = float('inf')
    for person in keypoints_with_scores:
        i = 0
        sum_distance = 0
        if len(select) <= 2:
            shaped = np.squeeze(np.multiply(person[:2], [y, x, 1]))
        else:
            shaped = np.squeeze(np.multiply(person, [y, x, 1]))
        for kp in shaped:
            ky, kx, _ = kp
            if len(select) > 2:
                sum_distance += abs(kx - select[i][1]) + abs(ky - select[i][0])
            else:
                sum_distance += abs(kx - select[1]) + abs(ky - select[0])
            i += 1
        if sum_distance < min_person:
            min_person = sum_distance
            right_person = person
    if min_person < 700:
        doChange = True
    else:
        doChange = False
    return right_person


def multiPose(select):
    global doChange
    doChange = True
    isFirstFrame = True
    cap = cv2.VideoCapture('vid18.mp4')

    while cap.isOpened():
        start_time = time.time()  # start time of the loop

        ret, frame = cap.read()

        # Resize image
        hi, wi, _ = frame.shape
        ratio = hi / wi

        wi = wi // 32
        wi *= 32
        wi = wi // 3

        if wi < 256:
            wi = 256

        hi = wi * ratio
        hi = hi // 32
        hi *= 32

        # there is a trade-off between Speed and Accuracy. (bigger images -> more accuracy -> low speed)
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), int(hi), int(wi))
        input_img = tf.cast(img, dtype=tf.int32)

        # Detection section
        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

        # detect the right person
        specific_person = detect_person(keypoints_with_scores, select)

        if isFirstFrame:
            isFirstFrame = False
            coords = []
            coords.append([int(specific_person[16][1] * frame.shape[1]),
                           int(specific_person[16][0] * frame.shape[0])])
            coords.append([int(specific_person[15][1] * frame.shape[1]),
                           int(specific_person[15][0] * frame.shape[0])])
            detectedLine = configureCoords(frame, coords)

        if doChange:
            # we render the right person we want to analyze
            y, x, _ = frame.shape
            select = np.squeeze(np.multiply(specific_person, [y, x, 1]))

        draw_connections(frame, specific_person, EDGES, 0)
        draw_keypoints(frame, specific_person, 0)

        # Render keypoints (all the people in the frame)
        # loop_through_people(frame, keypoints_with_scores, EDGES, 0.25)

        # fps
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get the size of the text
        size = cv2.getTextSize(str(1.0 / (time.time() - start_time)), font, 1, 2)

        # Calculate the position of the text
        x = int((img.shape[1] - size[0][0]/2))
        y = int((img.shape[0] + size[0][1]*2))

        # Add the text to the image
        cv2.putText(frame, str(1.0 / (time.time() - start_time)), (x, y), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(frame, (detectedLine[0][0], detectedLine[1][0]),
                 (detectedLine[0][1], detectedLine[1][1]), (255, 0, 0), 4)

        cv2.imshow('Multipose', frame)

        # check every 10 nanoseconds if the q is pressed to exits.
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture('vid18.mp4')
    if cap.isOpened():
        # read the first frame
        ret, frame = cap.read()
        cap.release()
        cv2.imshow('Selecting the person', frame)
        cv2.setMouseCallback('Selecting the person', mouse_callback)
        cv2.waitKey()

        # tracker = cv2.TrackerCSRT_create()
        #
        # # Select the person you want to track
        # person_roi = cv2.selectROI(frame)
        #
        # # Initialize the tracker with the person's ROI
        # tracker.init(frame, person_roi)
        # multiPose(tracker)