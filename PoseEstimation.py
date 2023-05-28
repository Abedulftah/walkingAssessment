import queue
import threading
import tkinter as tk

import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, structural_similarity
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
from Hough import *
from DepthEstimation import *
from MotionEstimation import *
from TestsResults import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

with tf.device('/GPU:0'):
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
real_time_size = (640, 480)


class PoseEstimation(threading.Thread):
    def __init__(self, PATH="video16_Trim.mp4", mainWindow=None, putDetectedLine=True):
        super(PoseEstimation, self).__init__()
        self.frame = None
        self.mainWindow = mainWindow
        self.putDetectedLine = putDetectedLine
        self.currentFrame = None
        self.PATH = PATH
        self.paused = False
        self.isWalking = False
        self.should_stop = threading.Event()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.destroyAllWindows()
            self.multiPose([y, x])

    def select_line(self, event, x, y, flags, param):
        global detectedLines
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.destroyAllWindows()
            if detectedLines is None:
                detectedLines = []
                detectedLines.append([int(x - 100), int(y), int(x + 100), int(y)])
            else:
                detectedLines.insert(0, [int(x - 100), int(y), int(x + 100), int(y)])

    def feetOnLine(self, frame1, frame2, startLine):
        coords = [int(startLine[0]), int(startLine[1] - 8), int(startLine[2]), int(startLine[3] + 20)]
        count, ind = 0, 0
        lst1, lst2 = [], []
        for x in range(coords[0], coords[2]+1):
            lst1.append([])
            lst2.append([])
            for y in range(coords[1], coords[3]+1):
                lst1[ind].append(frame1[y][x])
                lst2[ind].append(frame2[y][x])
            ind +=1
        frame1_temp = np.array(lst1)
        frame2_temp = np.array(lst2)

        # cv2.imshow('aaa', frame1_temp)
        frame1_temp = cv2.cvtColor(frame1_temp, cv2.COLOR_BGR2GRAY)
        frame1_temp = cv2.equalizeHist(frame1_temp)

        frame2_temp = cv2.cvtColor(frame2_temp, cv2.COLOR_BGR2GRAY)
        frame2_temp = cv2.equalizeHist(frame2_temp)

        (score, diff) = structural_similarity(frame1_temp, frame2_temp, gaussian_weights=True,
                                              use_sample_covariance=True, sigma=1.5, full=True)
        # cv2.imshow('sdsd', diff)
        # print(score)

        # for x in range(coords[0], coords[2]+1):
        #     for y in range(coords[1], coords[3]+1):
        #         if abs(int(frame1_temp[y][x]) - int(frame2_temp[y][x])) > 10:
        #             count += 1
        #     ind +=1
        # similarity = 10000 * count/(frame1.shape[0]*frame1.shape[1])
        return score < 0.57

    def draw_connections(self, frame, keypoints, edges, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 3, (0, 255, 0), -1)

    def loop_through_people(self, frame, keypoints_with_scores, edges, confidence_threshold):
        for person in keypoints_with_scores:
            self.draw_connections(frame, person, edges, confidence_threshold)
            self.draw_keypoints(frame, person, confidence_threshold)

    def find_person_keypoints(self, shaped, select, confidence_threshold):
        i = 0
        sum_distance = 0
        if len(select) > 2 and confidence_threshold:
            confidence_threshold = select[:, 2:]
            confidence_threshold = (np.max(confidence_threshold) + np.min(confidence_threshold)) / 2
        for kp in shaped:
            ky, kx, _ = kp
            if len(select) > 2 and select[i][2] >= confidence_threshold:
                sum_distance += abs(kx - select[i][1]) + abs(ky - select[i][0])
            elif len(select) <= 2:
                sum_distance += abs(kx - select[1]) + abs(ky - select[0])
            i += 1

        return sum_distance

    def detect_person(self, keypoints_with_scores, select):
        # there is one bug that when another person overlaps the right person.
        y, x, _ = self.frame.shape

        # to save the closest person of all the people that we found
        right_person = None
        right_personN = None
        min_person = float('inf')
        min_personN = float('inf')

        for person in keypoints_with_scores:
            if len(select) <= 2:
                shaped = np.squeeze(np.multiply(person[:2], [y, x, 1]))
            else:
                shaped = np.squeeze(np.multiply(person, [y, x, 1]))
            # find the right person with confidence

            sum_distance = self.find_person_keypoints(shaped, select, True)

            if sum_distance < min_person and sum_distance != 0:
                min_person = sum_distance
                right_person = person
            # find the right person without confidence
            sum_distance = self.find_person_keypoints(shaped, select, False)
            if sum_distance < min_personN:
                min_personN = sum_distance
                right_personN = person
        if right_person is None:
            return min_personN < 300, right_personN
        else:
            return min_person < 600, right_person

    def get_keypoints(self, frame, select):
        # Resize image
        hi, wi, di = frame.shape

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
        with tf.device('/GPU:0'):
            results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

        # detect the right person
        change_cord_rp, specific_person = self.detect_person(keypoints_with_scores, select)

        return keypoints_with_scores, img, change_cord_rp, specific_person

    def multiPose(self, select):
        global detectedLines
        isFirstFrame, frameCount = True, 0
        detectedLines = None
        xyxy = None
        rectangle_cord = []
        cap = cv2.VideoCapture(self.PATH)
        ret, frame = cap.read()
        movement_time = 0
        boundColor = (0, 0, 255)
        counter = 1
        frameQueue = queue.Queue()
        othersQueue = queue.Queue()
        walking_speed = 0
        secondTime = False
        passedFirst = False
        selectSaving = select

        start_time = (60 * 4) + 40
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time*1000)

        while cap.isOpened() or not frameQueue.empty():
            if self.should_stop.is_set():
                return 0
            while self.paused:
                pass

            while counter < 45:
                ret_temp, frame_temp = cap.read()
                frameQueue.put([ret_temp, frame_temp])
                lastBlockFrame = frame_temp
                keypoints_with_scores, img, change_cord_rp, specific_person = self.get_keypoints(frame, selectSaving)
                y, x, _ = frame.shape
                selectSaving = np.multiply(specific_person, [y, x, 1])
                othersQueue.put([keypoints_with_scores, img, change_cord_rp, specific_person])
                counter += 1
            if counter < 45:
                continue

            start_time = time.time()  # start time of the loop
            ret, frame1 = frameQueue.get()

            keypoints_with_scores, img, change_cord_rp, specific_person = othersQueue.get()

            if cap.isOpened():
                keypoints_with_scores1, img1, change_cord_rp1, specific_person1 = self.get_keypoints(lastBlockFrame,
                                                                                                     selectSaving)
                y, x, _ = frame.shape
                selectSaving = np.multiply(specific_person1, [y, x, 1])
                othersQueue.put([keypoints_with_scores1, img1, change_cord_rp1, specific_person1])

            coords = [[int(specific_person[16][1] * frame.shape[1]),
                       int(specific_person[16][0] * frame.shape[0])],
                      [int(specific_person[15][1] * frame.shape[1]),
                       int(specific_person[15][0] * frame.shape[0])]]
            if isFirstFrame:
                isFirstFrame = False
                self.firstFrame = frame.copy()
                detectedLines = configureCoords(frame, coords)
                if not self.putDetectedLine:
                    detectedLines = None

            if detectedLines is None or len(detectedLines) == 1:
                scale = 0.6
                out_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                cv2.imshow('Finding The Line', out_frame)
                cv2.setMouseCallback('Finding The Line', self.select_line)
                while detectedLines is None:
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):  # Press q to exit
                        exit()
                for row in range(2):
                    for col in range(4):
                        detectedLines[row][col] = int(detectedLines[row][col] / scale)

            if change_cord_rp:
                # we render the right person we want to analyze
                y, x, _ = frame.shape
                select = np.squeeze(np.multiply(specific_person, [y, x, 1]))
                print('in')

            # Calculating the distance of the current frame, and the 45'th frame from the end line.
            coords1 = [[int(specific_person1[16][1] * frame.shape[1]),
                       int(specific_person1[16][0] * frame.shape[0])],
                      [int(specific_person1[15][1] * frame.shape[1]),
                       int(specific_person1[15][0] * frame.shape[0])]]
            distance_from_line = min(coord_to_line_distance(coords[0], detectedLines[1]),
                   coord_to_line_distance(coords[1], detectedLines[1]))
            BlockFrameDistance = min(coord_to_line_distance(coords1[0], detectedLines[1]),
                   coord_to_line_distance(coords1[1], detectedLines[1]))
            # print(distance_from_line, "    ", BlockFrameDistance)
            # Now We can find if the person is moving forward by setting a threshold to the difference between them.
            dis_threshold = 50
            fine2 = False
            if lastBlockFrame is None or BlockFrameDistance <= 40 or abs(BlockFrameDistance - distance_from_line) > dis_threshold:
                fine2 = True

            fine = False
            if movement_time < (1.0 / (time.time() - start_time)) * 2:
                fine = True

            # why select and not specific person
            y, x, _ = frame.shape
            movement_time, xyxy, rectangle_cord, frame, self.isWalking = motionDetection(frame, frame1, np.multiply(specific_person, [y, x, 1]), fine,
                                                                                         boundColor, xyxy, movement_time, rectangle_cord, fine2)
            distance_from_start = 1000
            distance_from_line2 = 1000
            if specific_person[16][2] >= 0.25 and specific_person[15][2] >= 0.25:
                distance_from_line2 = max(coord_to_line_distance(coords[0], detectedLines[1]),
                       coord_to_line_distance(coords[1], detectedLines[1]))
                distance_from_start = min(coord_to_line_distance(coords[0], detectedLines[0]),
                       coord_to_line_distance(coords[1], detectedLines[0]))

            # passedFirst = False
            # print(distance_from_start)
            if 10 <= distance_from_start <= 20 and self.feetOnLine(self.firstFrame, frame, detectedLines[0]) and self.isWalking:
                print('on the first line')
                passedFirst = True

            if self.isWalking and distance_from_line2 > 50 and passedFirst and frameCount is not None:
                frameCount += 1
                print(frameCount)

            self.draw_connections(frame, specific_person, EDGES, 0.25)
            self.draw_keypoints(frame, specific_person, 0.25)

            # Render keypoints (all the people in the frame)
            # loop_through_people(frame, keypoints_with_scores, EDGES, 0.25)

            # fps
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Get the size of the text
            size = cv2.getTextSize(str(1.0 / (time.time() - start_time)), font, 1, 2)

            # Calculate the position of the text
            x = int((img.shape[1] - size[0][0] / 2))
            y = int((img.shape[0] + size[0][1] * 2))

            # Add the text to the image
            cv2.putText(frame, str(1.0 / (time.time() - start_time)), (x, y), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.line(frame, (int(detectedLines[0][0]), int(detectedLines[0][1])),
                     (int(detectedLines[0][2]), int(detectedLines[0][3])), (0, 255, 0), 4)
            if len(detectedLines) > 1:
                cv2.line(frame, (int(detectedLines[1][0]), int(detectedLines[1][1])),
                         (int(detectedLines[1][2]), int(detectedLines[1][3])), (0, 255, 0), 3)

            out_frame = cv2.resize(frame, (1350, 650))
            # self.mainWindow.update_image(out_frame)
            cv2.imshow('Video', out_frame)

            if distance_from_line2 <= 50 and passedFirst:
                if frameCount is not None:
                    walking_speed += 4 / (frameCount / 30)
                if secondTime:
                    walking_speed /= 2
                    secondTime = False
                    save_evaluation(self.PATH, walking_speed)

                print(walking_speed)
                frameCount = 0
                boundColor = (0, 255, 0)
                passedFirst = False
                secondTime = True
            elif passedFirst:
                boundColor = (255, 0, 0)
            else:
                boundColor = (0, 0, 255)

            frame = frame1

            if cap.isOpened():
                ret_temp, frame_temp = cap.read()
                frameQueue.put([ret_temp, frame_temp])
                lastBlockFrame = frame_temp
            else:
                lastBlockFrame = None
            # check every 10 nanoseconds if the q is pressed to exits.
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            movement_time += 1
        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.should_stop.set()

    def run(self):
        cap = cv2.VideoCapture(self.PATH)
        start_time = (60 * 4) + 40
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        if cap.isOpened():
            # read the first frame
            # global frame
            ret, self.frame = cap.read()
            cap.release()
            # print(cv2.meanStdDev(frame)[1][0][0], cv2.Laplacian(frame, cv2.CV_8UC1).var())
            cv2.imshow('Selecting the person', self.frame)
            cv2.setMouseCallback('Selecting the person', self.mouse_callback)
            cv2.waitKey()