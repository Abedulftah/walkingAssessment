import tensorflow as tf
import tensorflow_hub as hub
import cv2
import time
import numpy as np


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


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)


def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


cap = cv2.VideoCapture('5mins.mp4')
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

    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), int(hi), int(wi))
    input_img = tf.cast(img, dtype=tf.int32)

    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

    # Render keypoints
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.25)

    # fps
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2

    # Get the size of the text
    size = cv2.getTextSize(str(1.0 / (time.time() - start_time)), font, fontScale, thickness)

    # Calculate the position of the text
    x = int((img.shape[1] - size[0][0]/2))
    y = int((img.shape[0] + size[0][1]*2))

    # Add the text to the image
    cv2.putText(frame, str(1.0 / (time.time() - start_time)), (x, y), font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)

    cv2.imshow('Multipose', frame)

    # check every 10 nanoseconds if the q is pressed then exits.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

