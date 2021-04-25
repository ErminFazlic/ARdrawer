import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import math

class Hand():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, show=True):
        RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(RGBimg)

        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                if show:
                    self.mpDraw.draw_landmarks(img, landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPos(self, img, handNumber=0, show=True):

        posList = []

        if self.results.multi_hand_landmarks:
            height, width, channel = img.shape
            hand = self.results.multi_hand_landmarks[handNumber]
            for i, landmarks in enumerate(hand.landmark):
                lmX = int(landmarks.x * width)
                lmY = int(landmarks.y * height)
                posList.append([i, lmX, lmY])
        return posList

        #Works for either hand right now, 1 for open finger, 0 for closed
        #Index 0 = thumb
        #Index 1 = index finger
        #Index 2 = middle finger
        #Index 3 = ring finger
        #Index 4 = pinky finger
        #If both hands are open then the list will be double length and the first 5 are for the right hand, the last 5 for the left hand.
    def openFingers(self, img, handNumber=0, recurse = True):
        lmlist = self.findPos(img, handNumber)
        tipFingers = [4, 8, 12, 16, 20]
        openList = []
        if self.results.multi_handedness:
            if len(self.results.multi_handedness) == 2 and recurse:
                openList = self.openFingers(img, handNumber=1, recurse=False)
            lrHand = self.results.multi_handedness[handNumber].classification[0].label
        if len(lmlist) != 0:
            if lrHand == 'Right':
                if lmlist[tipFingers[0]][1] < lmlist[tipFingers[0] - 2][1]:
                    openList.append(1)
                else:
                    openList.append(0)
            else:
                if lmlist[tipFingers[0]][1] > lmlist[tipFingers[0] - 2][1]:
                    openList.append(1)
                else:
                    openList.append(0)

            for i in range(1, 5):
                if lmlist[tipFingers[i]][2] < lmlist[tipFingers[i] - 2][2]:
                    openList.append(1)
                else:
                    openList.append(0)

        return openList

    def countFingers(self, img, handNumber=0):
        return self.openFingers(img, handNumber=0).count(1)

    def isFist(self, img, handNumber=0):
        if self.results.multi_handedness:
            if self.countFingers(img, handNumber) == 0:
                return True
            else:
             return False

    def isThumbsDown(self, img, handNumber=0):
        lmlist = self.findPos(img, handNumber)
        if len(lmlist) != 0:
            for i in range(0, len(lmlist)):
                if lmlist[4][2] < lmlist[i][2]:
                    return False
            return True

    def isThumbsUp(self, img, handNumber=0):
        lmlist = self.findPos(img, handNumber)
        if len(lmlist) != 0:
            for i in range(0, len(lmlist)):
                if lmlist[4][2] > lmlist[i][2]:
                    return False
            return True

    def isOkSign(self, img, handNumber=0):
        lmlist = self.findPos(img, handNumber)
        if len(lmlist) != 0:
            x1, y1 = lmlist[4][1], lmlist[4][2]
            x2, y2 = lmlist[8][1], lmlist[8][2]
            length = math.hypot(x2 - x1, y2 - y1)
            if self.countFingers(img, handNumber) == 4 and length < 30:
                return True
            else:
                return False