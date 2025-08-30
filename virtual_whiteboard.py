import cv2
import numpy as np
import mediapipe as mp

# ---------------- CONFIG ----------------
CAM_INDEX = 0
FRAME_W = 640
FRAME_H = 480
AREA_THRESHOLD = 300

# Red pen HSV
H_MIN1 = 0
H_MAX1 = 10
H_MIN2 = 170
H_MAX2 = 180
S_MIN = 120
S_MAX = 255
V_MIN = 70
V_MAX = 255

# Drawing settings
DEFAULT_COLOR = (0,0,255)  # Red
ALT_COLOR = (0,255,0)      # Alternate color
THICKNESS = 5
ERASER_THICKNESS = 50
ERASER_ON = False

# ---------------- STATE ----------------
strokes = []
current_stroke = None
brush_color = DEFAULT_COLOR

# ---------------- MediaPipe Hands ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ---------------- CAPTURE ----------------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

def detect_gesture(hand_landmarks):
    # Finger tips and pip joints
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    pip = [3, 6, 10, 14, 18]
    fingers = []

    for t, p in zip(tips, pip):
        if t == 4:  # Thumb special case: check y axis for up
            if hand_landmarks.landmark[t].y < hand_landmarks.landmark[p].y:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if hand_landmarks.landmark[t].y < hand_landmarks.landmark[p].y:
                fingers.append(1)
            else:
                fingers.append(0)

    # Thumb Up: Thumb=1, all others=0
    if fingers[0]==1 and sum(fingers[1:])==0:
        return "THUMB_UP"
    # Victory / Peace Sign: Index=1, Middle=1, others=0
    elif fingers[1]==1 and fingers[2]==1 and fingers[0]==0 and sum(fingers[3:])==0:
        return "VICTORY"
    else:
        return "NONE"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Gesture detection
    gesture = "NONE"
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            gesture = detect_gesture(handLms)
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Gesture actions
    if gesture == "THUMB_UP":
        ERASER_ON = True
    elif gesture == "VICTORY":
        ERASER_ON = False
        brush_color = ALT_COLOR if brush_color == DEFAULT_COLOR else DEFAULT_COLOR
    else:
        ERASER_ON = False

    # Red pen detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([H_MIN1, S_MIN, V_MIN])
    upper_red1 = np.array([H_MAX1, S_MAX, V_MAX])
    lower_red2 = np.array([H_MIN2, S_MIN, V_MIN])
    upper_red2 = np.array([H_MAX2, S_MAX, V_MAX])

    mask = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask, mask2)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    # Contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    detected_point = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > AREA_THRESHOLD:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                detected_point = (cx, cy)

                color = (0,0,0) if ERASER_ON else brush_color
                thick = ERASER_THICKNESS if ERASER_ON else THICKNESS

                if current_stroke is None:
                    current_stroke = {'color': color, 'thickness': thick, 'points': [detected_point]}
                else:
                    current_stroke['points'].append(detected_point)

                cv2.circle(frame, detected_point, 8, color, -1)

    # Finalize stroke
    if detected_point is None and current_stroke is not None:
        if len(current_stroke['points']) > 0:
            strokes.append(current_stroke)
        current_stroke = None

    # Draw strokes
    canvas = np.zeros_like(frame)
    for s in strokes:
        pts = s['points']
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], s['color'], s['thickness'])
        if len(pts) == 1:
            cv2.circle(canvas, pts[0], s['thickness']//2, s['color'], -1)

    if current_stroke is not None:
        pts = current_stroke['points']
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], current_stroke['color'], current_stroke['thickness'])
        if len(pts) == 1:
            cv2.circle(canvas, pts[0], current_stroke['thickness']//2, current_stroke['color'], -1)

    # Overlay
    overlay = cv2.addWeighted(frame, 0.4, canvas, 0.6, 0)
    cv2.imshow("Virtual Whiteboard", overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        strokes.clear()
        current_stroke = None
        print("Canvas cleared")

cap.release()
cv2.destroyAllWindows()
