""" PoseEstimation class is to keep tracking the right person. by only letting the user clicking on the right person
at the first frame and in the next frames we'll continue track the person, here we use the MoveNet model by
tensorflow which also give us the ability to use the gpu, drawing the skeleton on the right person, detecting when
the right person passing the relevant lines by using structural similarity metric and here we use queue,
skimage.metrics, hough, DepthEstimation for detecting starting and ending line, MotionEstimation and DepthEstimation
is for detecting when the right person is walking, TestsResults is to save the speed estimation and getting the data
about specific patient, and the last one is ReadData for reading the videos from Google Drive. """

import queue
import threading
import tkinter as tk
from skimage.metrics import structural_similarity as ssim, structural_similarity
import tensorflow as tf
import tensorflow_hub as hub
import time
from Hough import *
from DepthEstimation import *
from MotionEstimation import *
from TestsResults import *
from ReadData import *

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
KEY_POINTS_NUMBER = 17


class PoseEstimation(threading.Thread):
    def __init__(self, PATH="video16_Trim.mp4", mainWindow=None, putDetectedLine=True, personFound=None):
        super(PoseEstimation, self).__init__()
        self.frame = None
        self.mainWindow = mainWindow
        self.putDetectedLine = putDetectedLine
        self.currentFrame = None
        self.PATH = PATH
        self.paused = False
        self.isWalking = False
        self.personFound = personFound
        self.should_stop = threading.Event()
        self.googleDrive = ReadData()
        self.T0T1T2 = None
        self.capIndex = 0
        self.usingGoogle = False

    # a method to take the coordinates of the selected person.
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
                detectedLines.insert(0, [int(x - 80), int(y), int(x + 80), int(y)])

    def feetOnLine(self, frame1, frame2, startLine):
        coords = [int(startLine[0]), int(startLine[1] - 8), int(startLine[2]), int(startLine[3] + 20)]
        count, ind = 0, 0
        lst1, lst2 = [], []
        for x in range(coords[0], coords[2] + 1):
            lst1.append([])
            lst2.append([])
            for y in range(coords[1], coords[3] + 1):
                lst1[ind].append(frame1[y][x])
                lst2[ind].append(frame2[y][x])
            ind += 1
        frame1_temp = np.array(lst1)
        frame2_temp = np.array(lst2)

        # cv2.imshow('aaa', frame1_temp)
        frame1_temp = cv2.cvtColor(frame1_temp, cv2.COLOR_BGR2GRAY)
        frame1_temp = cv2.equalizeHist(frame1_temp)

        frame2_temp = cv2.cvtColor(frame2_temp, cv2.COLOR_BGR2GRAY)
        frame2_temp = cv2.equalizeHist(frame2_temp)

        (score, diff) = structural_similarity(frame1_temp, frame2_temp, gaussian_weights=True,
                                              use_sample_covariance=True, sigma=1.5, full=True)

        return score < 0.57

    # a method to draw the lines between the coordinates of the detected person, taking only the coordinates that above
    # certain threshold.
    def draw_connections(self, frame, keypoints, edges, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # a method to draw the coordinates of the detected person, taking only the coordinates that above a certain
    # threshold
    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 3, (0, 255, 0), -1)

    # we used to method for debugging to draw all the detected people.
    def loop_through_people(self, frame, keypoints_with_scores, edges, confidence_threshold):
        for person in keypoints_with_scores:
            self.draw_connections(frame, person, edges, confidence_threshold)
            self.draw_keypoints(frame, person, confidence_threshold)

    # we need to take care of the case when the confidence of all coordinates is extremely low.
    # this method is to calculate the euclidean distance with some cases and returns the distance between the current
    # coordinate next frame coordinates.
    def find_person_keypoints(self, shaped, select, confidence_threshold):
        i, sum_distance, count = 0, 0, 0
        if len(select) > 2 and confidence_threshold:
            confidence_threshold = select[:, 2:]
            confidence_threshold = (np.max(confidence_threshold) + np.min(confidence_threshold)) / 2
        else:
            confidence_threshold = 0
        for kp in shaped:
            ky, kx, _ = kp
            # first case when the person is moving we take the Euclidian metric.
            if len(select) > 2 and select[i][2] >= confidence_threshold and self.isWalking:
                sum_distance += abs(kx - select[i][1]) + abs(ky - select[i][0])
                count += 1
            # second case when the person is not moving we take the Euclidian metric with a fine on the x coordinates.
            elif len(select) > 2 and select[i][2] >= confidence_threshold:
                sum_distance += abs(kx - select[i][1]) * 4 + abs(ky - select[i][0])
                count += 1
            # third case when the user provide us with the coordinates of the right person.
            elif len(select) <= 2:
                sum_distance += (abs(kx - select[1]) ** 2 + abs(ky - select[0]))
                count += 1
            i += 1

        return sum_distance, count

    # a method to track the right person.
    # returns specific person and a boolean which says if it is the right one.
    def detect_person(self, keypoints_with_scores, select):
        # there is one bug that when another person overlaps the right person.
        y, x, _ = self.frame.shape

        # to save the closest person of all the people that we found
        right_person = None
        min_person = float('inf')
        count = 0

        for person in keypoints_with_scores:
            if len(select) <= 2:
                shaped = np.squeeze(np.multiply(person[:2], [y, x, 1]))
            else:
                shaped = np.squeeze(np.multiply(person, [y, x, 1]))

            # find the right person with confidence
            sum_distance, count = self.find_person_keypoints(shaped, select, True)
            if sum_distance < min_person and sum_distance != 0:
                min_person = sum_distance + (10 * (KEY_POINTS_NUMBER - count))
                right_person = person


            if sum_distance == 0:
                # find the right person without confidence
                sum_distance, count = self.find_person_keypoints(shaped, select, False)
                if sum_distance < min_person:
                    min_person = sum_distance + (10 * (KEY_POINTS_NUMBER - count))
                    right_person = person

        print(min_person, count)
        return min_person < 600, right_person

    # a method to use the MoveNet model to get the coordinates of all the people we detected from the current frame.
    # returns the coordinates of all the people we found in the current frame, current frame, if the current person
    # is the right person, the specific person.
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
        keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, KEY_POINTS_NUMBER, 3))

        # detect the right person
        change_cord_rp, specific_person = self.detect_person(keypoints_with_scores, select)

        return keypoints_with_scores, img, change_cord_rp, specific_person

    def multiPose(self, select):
        global detectedLines
        self.mainWindow.personFound = select.copy()
        isFirstFrame, frameCount = True, 0
        detectedLines, xyxy, rectangle_cord = None, None, []
        if not self.usingGoogle:
            cap = cv2.VideoCapture(self.PATH)
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_time)
        else:
            cap = self.T0T1T2[self.capIndex]
        frame = self.frame
        movement_time = 0
        boundColor = (0, 0, 255)
        counter = 1
        frameQueue, othersQueue = queue.Queue(), queue.Queue()
        walking_speed, count_falses = 0, 0
        secondTime, passedFirst = False, False
        first_skeleton = select
        s = MotionDetection(0.25)

        # to get the start_time from excel.
        # no need for it if we have the ReadData class.
        # start_time = get_start_time(self.video_num, self.video_num_session)

        # to set the video to the starting time of the relevant section of the video.
        # cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        # video_writer = cv2.VideoWriter('output/data/output_100_T0.mp4',
        #                                cv2.VideoWriter_fourcc(*'mp4v'),
        #                                cap.get(cv2.CAP_PROP_FPS), (1350, 650))

        while cap.isOpened() or not frameQueue.empty():
            if self.should_stop.is_set():
                cv2.destroyAllWindows()
                return 0
            while self.paused:
                pass

            # the first 45 frame we put into a queue.
            while counter < 45:
                ret_temp, frame_temp = cap.read()
                frameQueue.put([ret_temp, frame_temp])
                lastBlockFrame = frame_temp
                keypoints_with_scores, img, change_cord_rp, specific_person = self.get_keypoints(frame_temp, select)
                othersQueue.put([keypoints_with_scores, img, change_cord_rp, specific_person])
                if change_cord_rp:
                    y, x, _ = frame.shape
                    select = np.squeeze(np.multiply(specific_person, [y, x, 1]))

                counter += 1

            if counter < 45:
                continue

            start_time = time.time()  # start time of the loop
            ret, frame1 = frameQueue.get()

            keypoints_with_scores, img, change_cord_rp, specific_person = othersQueue.get()

            # Maybe the if statement should be: if lastBlockFrame is not None.
            if cap.isOpened():
                keypoints_with_scores1, img1, change_cord_rp1, specific_person1 = self.get_keypoints(lastBlockFrame,
                                                                                                     select)
                othersQueue.put([keypoints_with_scores1, img1, change_cord_rp1, specific_person1])

                if change_cord_rp1:
                    y, x, _ = frame.shape
                    select = np.squeeze(np.multiply(specific_person1, [y, x, 1]))
                    count_falses = 0
                else:
                    count_falses += 1
            if count_falses >= 10:
                select = first_skeleton

            coords = [[int(specific_person[16][1] * frame.shape[1]),
                       int(specific_person[16][0] * frame.shape[0])],
                      [int(specific_person[15][1] * frame.shape[1]),
                       int(specific_person[15][0] * frame.shape[0])]]
            if isFirstFrame:
                isFirstFrame = False
                self.firstFrame = frame.copy()
                detectedLines = configureCoords(self.PATH, frame, coords)
                if not self.putDetectedLine:
                    detectedLines = None
                    self.putDetectedLine = True

            # if at least one of starting line, ending line is wrong we can click to pick them manual,
            # so this if is to restart the video.
            if not self.putDetectedLine:
                if self.usingGoogle:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.googleDrive.start_time[self.capIndex])
                cv2.destroyAllWindows()
                return

            scale = 0.6
            # If the algorithm didn't find any line, We let the user choose the end line,
            # and then the start line in the next if statement.
            if detectedLines is None:
                out_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                cv2.imshow('Choose end line!', out_frame)
                cv2.setMouseCallback('Choose end line!', self.select_line)
                while detectedLines is None:
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):  # Press q to exit
                        exit()
                for col in range(4):
                    detectedLines[0][col] = int(detectedLines[0][col] / scale)
                save_evaluation(self.PATH, detectedLines[0].copy(), 'End Line')

            # this will be True if the algorithm didn't find any line, or it only found the end line.
            if len(detectedLines) == 1:
                out_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                cv2.imshow('Choose start line!', out_frame)
                cv2.setMouseCallback('Choose start line!', self.select_line)
                while len(detectedLines) == 1:
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):  # Press q to exit
                        exit()
                for col in range(4):
                    detectedLines[0][col] = int(detectedLines[0][col] / scale)
                save_evaluation(self.PATH, detectedLines[0].copy(), 'Start Line')

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
            dis_threshold = 25
            fine2 = False
            if lastBlockFrame is None or BlockFrameDistance <= 40 or abs(
                    BlockFrameDistance - distance_from_line) > dis_threshold:
                fine2 = True

            # we don't need to check if we can notice the movement of the person at every frame we need to skip some,
            # if we detected movement we need to say it is moving for some frames even if we did
            # there is no movement.
            fine = False
            if movement_time < (1.0 / (time.time() - start_time)) * 2:
                fine = True

            # every thing is explained in the MotionEstimation class.
            y, x, _ = frame.shape
            movement_time, xyxy, rectangle_cord, frame, self.isWalking = s.motionDetection(frame, frame1,
                                                                                         np.multiply(specific_person,
                                                                                                     [y, x, 1]), fine,
                                                                                         boundColor, xyxy,
                                                                                         movement_time, rectangle_cord,
                                                                                         fine2,
                                                                                         walking_speed, secondTime)
            # here we check the distance from the lines.
            distance_from_start, distance_from_line2 = 1000, 1000
            if specific_person[16][2] >= 0.25 and specific_person[15][2] >= 0.25:
                distance_from_line2 = max(coord_to_line_distance(coords[0], detectedLines[1]),
                                          coord_to_line_distance(coords[1], detectedLines[1]))
                distance_from_start = min(coord_to_line_distance(coords[0], detectedLines[0]),
                                          coord_to_line_distance(coords[1], detectedLines[0]))

            # passedFirst = False
            # print(distance_from_start)
            # check if we passed the first line.
            moving_forward = distance_from_line - BlockFrameDistance
            if (-10 < distance_from_start < 10 or (
                    10 <= distance_from_start <= 30 and self.feetOnLine(self.firstFrame, frame, detectedLines[0]))) \
                    and self.isWalking and moving_forward >= dis_threshold and change_cord_rp:
                print('on the first line')
                passedFirst = True

            # we'll start counting the frames if we passed the first line.
            if self.isWalking and distance_from_line2 > 50 and passedFirst and frameCount is not None:
                frameCount += 1
                print(frameCount)

            # to draw the detected person.
            self.draw_connections(frame, specific_person, EDGES, 0.25)
            self.draw_keypoints(frame, specific_person, 0.25)

            # Render keypoints (all the people in the frame)
            # self.loop_through_people(frame, keypoints_with_scores, EDGES, 0.25)

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
            # video_writer.write(out_frame)
            # self.mainWindow.update_image(out_frame)
            cv2.imshow('Video', out_frame)

            # we calculate the speed of walking and saves it to the Excel.
            if distance_from_line2 <= 50 and passedFirst:
                if frameCount is not None:
                    walking_speed += 4 / (frameCount / 30)
                if secondTime:
                    walking_speed /= 2
                    secondTime = False
                    save_evaluation(self.PATH, walking_speed)
                    print(walking_speed)
                    break
                else:
                    secondTime = True

                print(walking_speed)
                frameCount = 0
                boundColor = (0, 255, 0)
                passedFirst = False
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
        # video_writer.release()
        cv2.destroyAllWindows()
        self.PATH = ""

    def stop(self):
        self.should_stop.set()

    def run(self):
        if self.PATH != "":
            cap = cv2.VideoCapture(self.PATH)

            folders = self.PATH.split('/')
            video_name = folders[-1].split('.')
            video_name = video_name[0].split('_')
            self.video_num = int(video_name[0])
            self.video_num_session = video_name[1]

            self.start_time = get_start_time(self.video_num, self.video_num_session) * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_time)
            if cap.isOpened():
                # read the first frame
                # global frame
                ret, self.frame = cap.read()
                cap.release()
                if self.personFound is not None:
                    self.multiPose(self.personFound)
                else:
                    cv2.imshow('Selecting the person', self.frame)
                    cv2.setMouseCallback('Selecting the person', self.mouse_callback)
                    cv2.waitKey()
        else:
            self.usingGoogle = True
            # we loop over all the patients.
            for patient in self.googleDrive.patients:
                if patient['name'] == '91':
                    continue
                self.T0T1T2 = self.googleDrive.googleDriveData(patient)
                self.capIndex = 0
                # we loop over the 3 videos or 4 video we got about the specific person (or what we got).
                while self.capIndex < len(self.T0T1T2):
                    if self.T0T1T2[self.capIndex] is None:
                        self.capIndex += 1
                        continue
                    if self.T0T1T2[self.capIndex].isOpened():
                        # read the first frame
                        # global frame
                        ret, self.frame = self.T0T1T2[self.capIndex].read()
                        if self.personFound is not None:
                            self.multiPose(self.personFound)
                        else:
                            # letting the user select the right person.
                            self.PATH = patient['name'] + '_T' + str(self.capIndex)
                            cv2.imshow('Selecting the person', self.frame)
                            cv2.setMouseCallback('Selecting the person', self.mouse_callback)
                            cv2.waitKey()
                        # this if means we detected at least on of the line wrong, so we'll let user select it manual.
                        if self.putDetectedLine:
                            self.capIndex += 1
