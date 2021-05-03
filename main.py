import cv2
import numpy as np

import HandTrackingModule as htm
import os


folder = 'Overlay'
imgList = os.listdir(folder)
overlayList = []
for img in imgList:
    image = cv2.imread(f'{folder}/{img}')
    overlayList.append(image)
sidebar = overlayList[0]
sidebar = sidebar[0:720, 0:115]

color = (255, 255, 255)
xp, yp = 0, 0
brushThickness = 15

capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

detector = htm.Hand()

canvasImg = np.zeros((720, 1280, 3), np.uint8)
canvasImg.fill(255)

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)

    detector.findHands(img, show=False)
    lmlist = detector.findPos(img, show=False)
    openList = detector.openFingers(img)

    if len(lmlist) != 0:
        indexX, indexY = lmlist[8][1], lmlist[8][2]
        if openList[1] == 1 and openList[2] == 0:
            print('Drawing')
            if color == (255, 255, 255):
                cv2.circle(img, (indexX, indexY), 20, color, cv2.FILLED)
            else:
                cv2.circle(img, (indexX, indexY), 10, color, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = indexX, indexY

            cv2.line(img, (xp, yp), (indexX, indexY), color, brushThickness)
            cv2.line(canvasImg, (xp, yp), (indexX, indexY), color, brushThickness+20)
            xp, yp = indexX, indexY

        elif openList[1] == 1 and openList[2] == 1:
            print('Selecting')
            xp, yp = 0, 0
            if indexX < 115:
                if 0 < indexY < 120:
                    sidebar = overlayList[1][0:720, 0:115]
                    color = (255, 113, 82)
                elif 120 < indexY < 240:
                    sidebar = overlayList[2][0:720, 0:115]
                    color = (22, 22, 255)
                elif 240 < indexY < 360:
                    sidebar = overlayList[3][0:720, 0:115]
                    color = (101, 226, 201)
                elif 360 < indexY < 480:
                    sidebar = overlayList[4][0:720, 0:115]
                    color = (55, 128, 0)
                elif 480 < indexY < 600:
                    sidebar = overlayList[5][0:720, 0:115]
                    color = (255, 255, 255)
                elif 600 < indexY < 720:
                    sidebar = overlayList[0][0:720, 0:115]

    imgGrey = cv2.cvtColor(canvasImg, cv2.COLOR_BGR2GRAY)
    _, imgInversed = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)

    img[0:720, 0:115] = sidebar
    #img = cv2.addWeighted(img, 0.5, canvasImg, 0.5, 0)
    cv2.imshow("ARdrawer", img)
    cv2.imshow("Canvas", canvasImg)
    cv2.waitKey(1)