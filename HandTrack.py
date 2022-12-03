# Created by Aryan Maurya

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # using mpDraw to draw the 21 points on hand

prevTime = 0    # previous time is 0
currTime = 0    # current time is 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting imagery to RGB
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)   -->     prints landmarks if hand is detected

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # considering each hand tracked
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
            # prints the id from 0 to 20 on x,y and z co-ordinates in decimal values.

                h, w, c = img.shape     # height,width and channel of image

                cx, cy = int(lm.x * w), int(lm.y * h)
                # central x and y axis -> taking integer value of x multiplied by width and int value of y multiplied by height
                print(id, cx, cy)
                # prints id no, cx & cy position

                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    # draws a small magenta circle on id 0 (i.e. on the wrist) of radius 15.

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)       # handLms --> single hand , Hand Connections to connect the dots


    currTime = time.time()      # stores the exact time during runtime
    fps = 1/(currTime-prevTime)     # formula to calculate frames per second
    prevTime = currTime         # previous time changes to current time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # inserting FPS monitor in img display as integer value.
    # (10,70) is the position for monitor, 3 and 3 are scale and thickness respectively.
    # (255,0,255) represents magenta color which is used for displaying fps.


    cv2.imshow("Image", img)  # Displays the original image
    cv2.waitKey(1)
