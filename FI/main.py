import cv2
import mediapipe as mp
import numpy as np
import time

link1 = 60
link2 = 40

def get_servo_angles(x, y, z):
    theta1 = np.arctan2(y, x)
    r_xy = np.sqrt(x**2 + y**2)
    r = np.sqrt(r_xy**2 + z**2)

    if r == 0 or r > (link1 + link2):
        return None

    cos_theta3 = (r**2 - link1**2 - link2**2) / (2 * link1 * link2)
    if abs(cos_theta3) > 1:
        return None

    theta3 = -np.arccos(cos_theta3)
    k1 = link1 + link2 * np.cos(theta3)
    k2 = link2 * np.sin(theta3)
    theta2 = np.arctan2(z, r_xy) - np.arctan2(k2, k1)

    base = int(np.degrees(theta1)) % 360
    shoulder = int(np.clip(np.degrees(theta2), 0, 180))
    elbow = int(np.clip(180 - abs(np.degrees(theta3)), 0, 180))

    return base, shoulder, elbow

def get_middle_finger_length(landmarks, w, h):
    mcp = np.array([landmarks[9].x * w, landmarks[9].y * h])
    tip = np.array([landmarks[12].x * w, landmarks[12].y * h])
    return np.linalg.norm(mcp - tip)

def map_openness_dynamic(thumb, index, mid_len, w, h):
    p1 = np.array([thumb.x * w, thumb.y * h])
    p2 = np.array([index.x * w, index.y * h])
    dist = np.linalg.norm(p1 - p2)
    norm = dist / mid_len
    return int(np.clip((norm - 0.15) / (2 - 0.35), 0, 1) * 100)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.9, max_num_hands=1)
cap = cv2.VideoCapture(0)

alpha = 0.5
prev_x, prev_y, prev_z, prev_open = 0, 0, 0, 0
z_locked = False
z_reference = 0
z_lock_time = None
z_range = 0.05
middle_len_reference = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    box_w, box_h = int(w * 0.8), int(h * 0.8)
    start_x = (w - box_w) // 2
    start_y = (h - box_h) // 2
    end_x = start_x + box_w
    end_y = start_y + box_h
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            landmarks = hand.landmark
            thumb = landmarks[4]
            index = landmarks[8]

            thumb_xyz = (thumb.x * w, thumb.y * h, thumb.z)
            index_xyz = (index.x * w, index.y * h, index.z)
            mid_x = (thumb_xyz[0] + index_xyz[0]) / 2
            mid_y = (thumb_xyz[1] + index_xyz[1]) / 2
            mid_z = (thumb_xyz[2] + index_xyz[2]) / 2

            middle_len = get_middle_finger_length(landmarks, w, h)
            if middle_len_reference is None:
                middle_len_reference = middle_len
                print(f"📏 Middle finger locked at: {middle_len_reference:.2f}")

            openness = map_openness_dynamic(thumb, index, middle_len, w, h)


            x = alpha * mid_x + (1 - alpha) * prev_x
            y = alpha * mid_y + (1 - alpha) * prev_y
            z_raw = alpha * mid_z + (1 - alpha) * prev_z
            open_smooth = alpha * openness + (1 - alpha) * prev_open

            prev_x, prev_y, prev_z, prev_open = x, y, z_raw, open_smooth

            if not z_locked:
                if z_lock_time is None:
                    z_lock_time = time.time()
                    print("🕒 Hold hand still for 3s to lock Z")
                elif time.time() - z_lock_time >= 3:
                    z_reference = z_raw
                    z_locked = True
                    print(f"✅ Z locked at: {z_reference:.4f}")
                else:
                    print(f"⏳ Locking in... {3 - int(time.time() - z_lock_time)}s")
                continue

            dz = z_raw - z_reference
            dz_clamped = np.clip((dz + z_range) / (2 * z_range), 0, 1)
            z_mapped = dz_clamped * 100

            x_clamped = np.clip((x - start_x) / box_w, 0, 1)
            y_clamped = np.clip((y - start_y) / box_h, 0, 1)
            x_percent = x_clamped * 100
            y_percent = y_clamped * 100

            result = get_servo_angles(x_percent, y_percent, z_mapped)
            if result:
                a, b, c = result
                print(f"x: {a}, y: {b}, z: {c}, openness: {int(open_smooth)}% Current middle finger length: {middle_len:.2f}")
            else:
                print("❌ Out of range")

            cv2.line(frame, (int(thumb_xyz[0]), int(thumb_xyz[1])),
                            (int(index_xyz[0]), int(index_xyz[1])), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 8, (255, 0, 0), -1)

            # Draw yellow line along middle finger
            mcp = landmarks[9]
            pip = landmarks[10]
            dip = landmarks[11]
            tip = landmarks[12]
            points = [mcp, pip, dip, tip]
            for i in range(len(points)-1):
                p1 = (int(points[i].x * w), int(points[i].y * h))
                p2 = (int(points[i+1].x * w), int(points[i+1].y * h))
                cv2.line(frame, p1, p2, (0, 255, 255), 2)

    cv2.imshow("Hand Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        z_locked = False
        z_lock_time = None
        print("🔄 Z lock reset")

cap.release()
cv2.destroyAllWindows()
