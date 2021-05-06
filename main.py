import ntpath

import cv2
import numpy as np

import HandTrackingModule as htm
import os
import time


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
saving = False
clearing = False
clear_time = time.time()
save_time = time.time()
save_path = 'screenshots/'

capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

detector = htm.Hand()

canvasImg = np.zeros((720, 1280, 3), np.uint8)
canvasImg.fill(255)

overlayDrawing = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)

    detector.findHands(img, show=False)
    lmlist = detector.findPos(img, show=False)
    openList = detector.openFingers(img)

    if len(lmlist) != 0:
        indexX, indexY = lmlist[8][1], lmlist[8][2]
        if openList[1] == 1 and openList[2] == 0:
            #print('Drawing')
            if color == (255, 255, 255):
                cv2.circle(img, (indexX, indexY), 20, color, cv2.FILLED)
            else:
                cv2.circle(img, (indexX, indexY), 10, color, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = indexX, indexY

            #cv2.line(img, (xp, yp), (indexX, indexY), color, brushThickness+20)

            if color == (255, 255, 255):
                cv2.line(overlayDrawing, (xp, yp), (indexX, indexY), (0,0,0), brushThickness + 40)
                cv2.line(canvasImg, (xp, yp), (indexX, indexY), color, brushThickness + 40)
            else:
                cv2.line(overlayDrawing, (xp, yp), (indexX, indexY), color, brushThickness + 20)
                cv2.line(canvasImg, (xp, yp), (indexX, indexY), color, brushThickness + 20)

            xp, yp = indexX, indexY

        elif openList[1] == 1 and openList[2] == 1:
            #print('Selecting')
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
                elif 600 < indexY:
                    sidebar = overlayList[0][0:720, 0:115]
                    canvasImg.fill(255)
                    overlayDrawing.fill(0)




    imgGrey = cv2.cvtColor(overlayDrawing, cv2.COLOR_BGR2GRAY)
    _, imgInversed = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInversed = cv2.cvtColor(imgInversed, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInversed)
    img = cv2.bitwise_or(img, overlayDrawing)

    if detector.isThumbsDown(img):
        xp, yp = 0, 0
        if not clearing:
            clearing = True
            clear_time = time.time()
        elif clear_time-time.time() < -2:
            #print("clear")
            canvasImg.fill(255)
            overlayDrawing.fill(0)
            clearing = False
    else:
        clearing = False

    if detector.isThumbsUp(img):
        xp, yp = 0, 0
        if not saving:
            saving = True
            save_time = time.time()
        elif save_time-time.time() < -2:
            #print("save")
            if os.path.isdir(save_path) == False:
                os.mkdir(save_path)
            cv2.imwrite(save_path + time.strftime("%Y-%m-%d-%H.%M.%S") + '.jpg', img)
            cv2.imwrite(save_path + 'Canvas-'+time.strftime("%Y-%m-%d-%H.%M.%S") + '.jpg', canvasImg)
            cv2.putText(img, 'Saving', (500, 400), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 2)

            time.sleep(2)
            saving = False

    else:
        saving = False


    img[0:720, 0:115] = sidebar
    #img = cv2.addWeighted(img, 0.5, canvasImg, 0.5, 0)
    cv2.imshow("ARdrawer", img)
    cv2.imshow("Canvas", canvasImg)
    cv2.waitKey(1)