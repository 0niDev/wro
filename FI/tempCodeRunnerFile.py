import cv2
import mediapipe as mp
import numpy as np
import math
import serial


send_to_serial = False
if send_to_serial:
    ser = serial.Serial('COM3', 9600)  # Change COM3 to your port

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.9, max_num_hands=1)
cap = cv2.VideoCapture(0)

# Distance function
def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Inverse Kinematics for 2DOF arm + base
def inverse_kinematics(x, y, L1=100, L2=100):
    dist = math.sqrt(x**2 + y**2)
    dist = min(dist, L1 + L2 - 1e-6)  # clamp to max reachable
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    theta2 = math.acos(np.clip(cos_theta2, -1.0, 1.0))
    k1 = L1 + L2 * math.cos(theta2)
    k2 = L2 * math.sin(theta2)
    theta1 = math.atan2(y, x) - math.atan2(k2, k1)
    return math.degrees(theta1), math.degrees(theta2)

# Map hand openness to 0–100%
def map_openness(raw_dist, min_dist=50, max_dist=170):
    raw_dist = np.clip(raw_dist, min_dist, max_dist)
    openness = (raw_dist - min_dist) / (max_dist - min_dist)
    return int(openness * 100)

# Smoothing setup
alpha = 0.3
prev_s1, prev_s2, prev_s3, prev_s4 = 90, 90, 90, 50  # neutral

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            landmarks = hand.landmark
            thumb = landmarks[4]
            index = landmarks[8]

            # 3D midpoint of thumb and index
            thumb_xyz = (thumb.x * w, thumb.y * h, thumb.z)
            index_xyz = (index.x * w, index.y * h, index.z)
            mid_x = (thumb_xyz[0] + index_xyz[0]) / 2
            mid_y = (thumb_xyz[1] + index_xyz[1]) / 2
            mid_z = (thumb_xyz[2] + index_xyz[2]) / 2

            # Map to arm space
            x = mid_x - w / 2
            y = h - mid_y
            base_angle = np.interp(x, [-200, 200], [0, 180])
            shoulder, elbow = inverse_kinematics(x, y)
            claw = map_openness(get_distance(thumb_xyz, index_xyz))

            # Smooth
            servo1 = int(alpha * base_angle + (1 - alpha) * prev_s1)
            servo2 = int(alpha * shoulder + (1 - alpha) * prev_s2)
            servo3 = int(alpha * elbow + (1 - alpha) * prev_s3)
            servo4 = int(alpha * claw + (1 - alpha) * prev_s4)

            prev_s1, prev_s2, prev_s3, prev_s4 = servo1, servo2, servo3, servo4
            if send_to_serial:
                data = f"&{servo1}&{servo2}&{servo3}&{servo4}\n"
                ser.write(data.encode())
            print(f"Base: {servo1}, Shoulder: {servo2}, Elbow: {servo3}, Claw: {servo4}%")

            cv2.line(frame,
                     (int(thumb_xyz[0]), int(thumb_xyz[1])),
                     (int(index_xyz[0]), int(index_xyz[1])),
                     (0, 255, 0), 3)

    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
