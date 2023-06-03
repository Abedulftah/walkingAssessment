import cv2


def motionDetection(frame1, frame2, specific_person, fine, boundColor, xyxy, movement_time, rectangle_cord, fine2=True, walking_speed=0, secondTime=False):
    noise = cv2.meanStdDev(frame1)[1][0][0]
    isWalking = False

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
        if fine2 and (fine or rectangle_cord[4]):
            cv2.rectangle(frame1, (rectangle_cord[0], rectangle_cord[1]),
                          (rectangle_cord[0] + rectangle_cord[2], rectangle_cord[1] + rectangle_cord[3]), boundColor,
                          2)
            if walking_speed == 0 or secondTime:
                cv2.putText(frame1, "Status: {}".format('Movement'), (rectangle_cord[0], rectangle_cord[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)
            elif not secondTime:
                cv2.putText(frame1, "Status: Movemnt, speed: {}".format(walking_speed), (rectangle_cord[0], rectangle_cord[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)
            isWalking = True

        if rectangle_cord[4]:
            movement_time = 0
            rectangle_cord[4] = False

    return movement_time, xyxy, rectangle_cord, frame1, isWalking

