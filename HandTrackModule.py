# Creating a module for HandTrack which we can use later

import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):     # parameters taken from the Hands() module
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity         # thank you stackoverflow
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # using mpDraw to draw the 21 points on hand


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting imagery to RGB
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)   -->     prints landmarks if hand is detected

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # considering each hand tracked
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)       # handLms --> single hand , Hand Connections to connect the dots
        return img


    def findPosition(self, img, handNo=0, draw=True):

        lmList = []         # to collect the landmark positions

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                # prints the id from 0 to 20 on x,y and z co-ordinates in decimal values.

                h, w, c = img.shape  # height,width and channel of image

                cx, cy = int(lm.x * w), int(lm.y * h)
                # central x and y axis -> taking integer value of x multiplied by width and int value of y multiplied by height
                # print(id, cx, cy)
                # prints id no, cx & cy position
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return lmList



def main():
    prevTime = 0  # previous time is 0
    currTime = 0  # current time is 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)   # draw=False means to hide the points we drawn on hands
        if len(lmList) != 0:
            print(lmList[4])


        currTime = time.time()  # stores the exact time during runtime
        fps = 1 / (currTime - prevTime)  # formula to calculate frames per second
        prevTime = currTime  # previous time changes to current time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        # inserting FPS monitor in img display as integer value.
        # (10,70) is the position for monitor, 3 and 3 are scale and thickness respectively.
        # (255,0,255) represents magenta color which is used for displaying fps.

        cv2.imshow("Image", img)  # Displays the original image
        cv2.waitKey(1)




if __name__ == "__main__":
    main()




'''
              y-axis
        value approx 0 on top
                |
                |
                |
                |
 150<-------------------------->640     x-axis
                |
                |
                |
                |
            approx 480
            
            
    this is a 640x480 img    


'''
