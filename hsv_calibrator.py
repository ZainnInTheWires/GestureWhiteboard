import cv2
import numpy as np

def nothing(x):
    pass

# Webcam start
cap = cv2.VideoCapture(0)

# Trackbars window
cv2.namedWindow('Calibrator')
cv2.createTrackbar('H_min','Calibrator',0,179,nothing)
cv2.createTrackbar('H_max','Calibrator',179,179,nothing)
cv2.createTrackbar('S_min','Calibrator',50,255,nothing)
cv2.createTrackbar('S_max','Calibrator',255,255,nothing)
cv2.createTrackbar('V_min','Calibrator',50,255,nothing)
cv2.createTrackbar('V_max','Calibrator',255,255,nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of trackbars
    hmin = cv2.getTrackbarPos('H_min','Calibrator')
    hmax = cv2.getTrackbarPos('H_max','Calibrator')
    smin = cv2.getTrackbarPos('S_min','Calibrator')
    smax = cv2.getTrackbarPos('S_max','Calibrator')
    vmin = cv2.getTrackbarPos('V_min','Calibrator')
    vmax = cv2.getTrackbarPos('V_max','Calibrator')

    lower = np.array([hmin,smin,vmin])
    upper = np.array([hmax,smax,vmax])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', res)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
