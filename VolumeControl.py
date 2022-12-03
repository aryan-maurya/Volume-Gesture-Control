# Created by Aryan Maurya

import cv2
import time
import numpy as np
import HandTrackModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#####################
wCam, hCam = 640, 480      # defining height and width of camera
####################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prevTime = 0

detector = htm.handDetector(detectionCon=0.7)          # increasing the detection confidence so we are sure that it's a hand



devices = AudioUtilities.GetSpeakers()                  # pycaw on github to control volume
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()          # thankyou Andre Miras
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4], lmList[8])        # 4 for the tip of thumb, 8 for the tip of index finger

        x1, y1 = lmList[4][1], lmList[4][2]     # element 1 and 2 of thumb and index as x & y
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2     # finding the center of axis


        cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)    # marking a circle on thumb and index tips
        cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)         # drawing a line to connect both
        cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)    # finding the center of line


        length = math.hypot(x2 - x1, y2 - y1)       # taking hypotenuse
        print(length)                               # print hypo --> value increases/decreases, considering (50-300)


        # Hand Range 50 to 300
        # Volume Range -65 to 0

        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)


        if length < 50:
            cv2.circle(img, (cx, cy), 6, (0, 255, 0), cv2.FILLED)       # the center becomes green when fingers touch

    cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)            # rectangle for showing volume
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)   # volume changing bar
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)


    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
