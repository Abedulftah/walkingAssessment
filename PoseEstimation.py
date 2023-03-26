import cv2
import tensorflow as tf
import tensorflow_hub as hub
import time
from Hough import *
from DepthEstimation import *

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
rectangle_cord = []
PATH = '5mins.mp4'


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.destroyAllWindows()
        multiPose([y, x])


def select_line(event, x, y, flags, param):
    global detectedLine
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.destroyAllWindows()
        detectedLine = [[int(x-100), int(x+100)], [int(y), int(y)]]


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


def find_person_keypoints(shaped, select, confidence_threshold):
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


def detect_person(keypoints_with_scores, select):
    # there is one bug that when another person overlaps the right person.
    global change_cord_rp
    y, x, _ = frame.shape

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

        sum_distance = find_person_keypoints(shaped, select, True)

        if sum_distance < min_person and sum_distance != 0:
            min_person = sum_distance
            right_person = person
        # find the right person without confidence
        sum_distance = find_person_keypoints(shaped, select, False)
        if sum_distance < min_personN:
            min_personN = sum_distance
            right_personN = person

    if min_person < 600:
        change_cord_rp = True
    else:
        change_cord_rp = False
    # print(min_person, min_personN)
    if right_person is None:
        if min_personN < 300:
            change_cord_rp = True
        else:
            change_cord_rp = False
        return right_personN
    return right_person


def is_moving_forward(xyxy, center_x, center_y):

    # Calculate motion vector
    motion_vector1 = (center_x - xyxy[0], center_y - xyxy[1])  # Determine direction of motion
    if abs(motion_vector1[0]) > abs(motion_vector1[1]):
        if motion_vector1[0] > 0:
            print('Object moved right')
        else:
            print('Object moved left')
        return False
    else:
        if motion_vector1[1] > 0:
            print('Object moved down')
        else:
            print('Object moved up')
        return True

def motionDetection(frame1, frame2, specific_person, fine, boundColor):
    global movement_time, xyxy
    noise = cv2.meanStdDev(frame1)[1][0][0]

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = gray
    if noise < 50:
        blur = cv2.GaussianBlur(blur, (5, 5), 1.5)
    else:
        blur = cv2.Laplacian(blur, cv2.CV_8UC1)

    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, None, iterations=5)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        moving = True
        moving_forward = True
        (x, y, w, h) = cv2.boundingRect(contour)
        if specific_person[16][0] < y or specific_person[16][0] > y + h or specific_person[16][1] < x or \
                specific_person[16][1] > x + w or \
                specific_person[15][0] < y or specific_person[15][0] > y + h or specific_person[15][1] < x or \
                specific_person[15][1] > x + w:
            moving = False

        if xyxy is None:
            xyxy = (specific_person[16][1], specific_person[16][0], specific_person[15][1], specific_person[15][0])
        elif moving:
            # moving_forward = is_moving_forward(xyxy, x + w // 2, y + h // 2)
            pass

        if movement_time % 2 == 0:
            xyxy = (specific_person[16][1], specific_person[16][0], specific_person[15][1], specific_person[15][0])
        if moving and moving_forward:
            rectangle_cord.clear()
            rectangle_cord.append(x)
            rectangle_cord.append(y)
            rectangle_cord.append(w)
            rectangle_cord.append(h)
            rectangle_cord.append(True)
            break

    if len(rectangle_cord) > 0:
        if fine or rectangle_cord[4]:
            cv2.rectangle(frame1, (rectangle_cord[0], rectangle_cord[1]),
                          (rectangle_cord[0] + rectangle_cord[2], rectangle_cord[1] + rectangle_cord[3]), boundColor,
                          2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (rectangle_cord[0], rectangle_cord[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
        if rectangle_cord[4]:
            movement_time = 0
            rectangle_cord[4] = False

    return frame1


def multiPose(select):
    global change_cord_rp, movement_time, xyxy, detectedLine
    change_cord_rp = True
    isFirstFrame = True
    detectedLine = None
    xyxy = None
    cap = cv2.VideoCapture(PATH)
    ret, frame = cap.read()
    movement_time = 0
    boundColor = (0, 0, 255)

    while cap.isOpened():
        start_time = time.time()  # start time of the loop
        ret, frame1 = cap.read()

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
        specific_person = detect_person(keypoints_with_scores, select)
        coords = [[int(specific_person[16][1] * frame.shape[1]),
                   int(specific_person[16][0] * frame.shape[0])], [int(specific_person[15][1] * frame.shape[1]),
                                                                   int(specific_person[15][0] * frame.shape[0])]]
        if isFirstFrame:
            isFirstFrame = False
            detectedLine = configureCoords(frame, coords)

        if change_cord_rp:
            # we render the right person we want to analyze
            y, x, _ = frame.shape
            select = np.squeeze(np.multiply(specific_person, [y, x, 1]))

        fine = False
        if movement_time < (1.0 / (time.time() - start_time)) * 2:
            fine = True

        # why select and not specific person
        frame = motionDetection(frame, frame1, select, fine, boundColor)
        draw_connections(frame, specific_person, EDGES, 0.25)
        draw_keypoints(frame, specific_person, 0.25)

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

        if detectedLine is None:
            scale = 0.7
            out_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            cv2.imshow('Finding The Line', out_frame)
            cv2.setMouseCallback('Finding The Line', select_line)
            while detectedLine is None:
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):  # Press q to exit
                    exit()
            detectedLine[0][0] = int(detectedLine[0][0] / scale)
            detectedLine[0][1] = int(detectedLine[0][1] / scale)
            detectedLine[1][0] = int(detectedLine[1][0] / scale)
            detectedLine[1][1] = int(detectedLine[1][1] / scale)

        cv2.line(frame, (detectedLine[0][0], detectedLine[1][0]),
                 (detectedLine[0][1], detectedLine[1][1]), (255, 0, 0), 4)

        out_frame = cv2.resize(frame, (1350, 650))
        cv2.imshow('Multipose', out_frame)

        if min(coord_to_line_distance(coords[0], detectedLine),
               coord_to_line_distance(coords[1], detectedLine)) <= 0:
            boundColor = (0, 255, 0)

        frame = frame1
        # check every 10 nanoseconds if the q is pressed to exits.
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        movement_time += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture(PATH)
    if cap.isOpened():
        # read the first frame
        ret, frame = cap.read()
        cap.release()
        # print(cv2.meanStdDev(frame)[1][0][0], cv2.Laplacian(frame, cv2.CV_8UC1).var())
        cv2.imshow('Selecting the person', frame)
        cv2.setMouseCallback('Selecting the person', mouse_callback)
        cv2.waitKey()