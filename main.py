import cv2
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

color = (0, 0, 0)

capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

detector = htm.Hand()

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)

    detector.findHands(img, show=False)
    lmlist = detector.findPos(img, show=False)

    if len(lmlist) != 0:
        indexX, indexY = lmlist[8][1], lmlist[8][2]
        if detector.openFingers(img)[1] == 1 and detector.countFingers(img) == 1:
            print('Drawing')
            cv2.circle(img, (indexX, indexY), 20, color, cv2.FILLED)
        elif detector.openFingers(img)[1] == 1 and detector.countFingers(img) > 1:
            print('Selecting')
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


    img[0:720, 0:115] = sidebar
    cv2.imshow("ARdrawer", img)
    cv2.waitKey(1)