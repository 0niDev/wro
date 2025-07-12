import cv2
import numpy as np
from collections import deque

def set_motor_power(fl, fr, bl, br, side, forward):
    pass

def move(x, y):
    fl = y + x
    fr = y - x
    bl = y - x
    br = y + x
    max_val = max(abs(fl), abs(fr), abs(bl), abs(br), 1)
    fl /= max_val
    fr /= max_val
    bl /= max_val
    br /= max_val
    fl_int = round((fl + 1) / 2 * 255)
    fr_int = round((fr + 1) / 2 * 255)
    bl_int = round((bl + 1) / 2 * 255)
    br_int = round((br + 1) / 2 * 255)
    set_motor_power(fl_int, fr_int, bl_int, br_int, x, y)

cap = cv2.VideoCapture(0)

side_history = deque(maxlen=5)
forward_history = deque(maxlen=5)

def calc_move(h, height):
    ratio = h / height
    if ratio > 0.5:
        side = 1.0
        forward = 0.0
    elif ratio < 0.1:
        side = 0
        forward = 1.0
    else:
        side = min(ratio * 2, 1.0)
        forward = max(1 - ratio * 1.5, 0.3)
    return side, forward

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]
    marker_x = width // 2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 150])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_box = None
    if red_contours:
        red_contours = sorted(red_contours, key=cv2.contourArea, reverse=True)
        if cv2.contourArea(red_contours[0]) > 300:
            red_box = cv2.boundingRect(red_contours[0])
            x, y, w, h = red_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    green_box = None
    if green_contours:
        green_contours = sorted(green_contours, key=cv2.contourArea, reverse=True)
        if cv2.contourArea(green_contours[0]) > 300:
            green_box = cv2.boundingRect(green_contours[0])
            x, y, w, h = green_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if red_box and not green_box:
        _, _, _, h = red_box
        side, forward = calc_move(h, height)
        side_history.append(-side)
        forward_history.append(forward)
        print("\033[91mRed block detected: moving left\033[0m")  # red text
    elif green_box and not red_box:
        _, _, _, h = green_box
        side, forward = calc_move(h, height)
        side_history.append(side)
        forward_history.append(forward)
        print("\033[92mGreen block detected: moving right\033[0m")  # green text
    elif red_box and green_box:
        _, _, _, red_h = red_box
        _, _, _, green_h = green_box
        if red_h > green_h:
            side, forward = calc_move(red_h, height)
            side_history.append(-side)
            forward_history.append(forward)
            print("\033[91mRed block priority: moving left\033[0m")
        else:
            side, forward = calc_move(green_h, height)
            side_history.append(side)
            forward_history.append(forward)
            print("\033[92mGreen block priority: moving right\033[0m")
    else:
        side_history.append(0)
        forward_history.append(1)
        print("No block detected, moving forward")

    avg_side = sum(side_history) / len(side_history)
    avg_forward = sum(forward_history) / len(forward_history)

    move(avg_side, avg_forward)

    cv2.line(frame, (marker_x, 0), (marker_x, height), (255, 0, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
