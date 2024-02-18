import time
import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector

wCam, hCam = 1080, 800
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
offset = 200

cap = cv2.VideoCapture(0) 
cap.set(3, wCam) 
cap.set(4, hCam)  
pTime = 0

detector = HandDetector(detectionCon=0.85) 

def makePosition(img):
    lmList = []
    if detector.results.multi_hand_landmarks:
        myHand = detector.results.multi_hand_landmarks[0]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
    return lmList

drawThickness = 15
prev_coords = {4:(0, 0), 8:(0, 0)}

def map_line(cv, landmarks, index, color):
    x, y = landmarks[index][1:]
    if prev_coords[index] == (0, 0):
        prev_coords[index] = (x, y)
        
    cv2.line(cv, (wCam + offset - x, y), (wCam + offset - prev_coords[index][0], prev_coords[index][1]), 
             color, drawThickness)
    
    prev_coords[index] = (x, y)
    
def map_between(cv, index1, index2, color):
    if prev_coords[index1] != (0, 0) and prev_coords[index2] != (0, 0):
        cv2.line(cv, (wCam + offset - prev_coords[index1][0], prev_coords[index1][1]), 
                 (wCam + offset - prev_coords[index2][0], prev_coords[index2][1]),
                 color, 2)
        return math.sqrt((prev_coords[index1][0] - prev_coords[index2][0]) ** 2
                         + (prev_coords[index1][1] - prev_coords[index2][1]) ** 2)
    return -1

dist_between = 0
dist_threshold = 75

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    hands = detector.findHands(imgRGB)
    img = cv2.flip(img, 1)
    landmarks = makePosition(img)
    
    if len(landmarks) != 0:
        map_line(img if dist_between < 0 or dist_between > dist_threshold else canvas, landmarks, 8, (240, 200, 50))
        map_line(img, landmarks, 4, (200, 240, 50))
        dist_between = map_between(img, 4, 8, (0, 0, 255) if dist_between < 0 or dist_between > dist_threshold else (0, 255, 0))

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (200, 200, 0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()