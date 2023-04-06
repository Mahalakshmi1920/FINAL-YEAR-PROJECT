import cv2
import cvzone
import time

from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

star_time = time.time()

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5,color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        # check 10s check 
        end_time = time.time() - star_time
        print(end_time) # Only verify we can remove

        if ratioAvg < 35 and counter == 0:
            star_time = time.time()
            blinkCounter += 1
            color = (0,200,0)
            counter = 1

        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255,0, 255)

        # Condition to check more than 10 seconds eyes not blinked
        if end_time > 10:
            break

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100),
                           colorR=color)

        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (500, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (500, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("OutputVDO", imgStack)
    # Break gracefully
    q = cv2.waitKey(30) & 0xff # press ESC to exit
    if q == 27 or cv2.getWindowProperty('OutputVDO', 0)<0:
        break
    cv2.waitKey(25)
if blinkCounter > 0:
    cvzone.putTextRect(img, "Eyes Closed. Press 'c' to continue", (50, 200), colorR=(0, 0, 255))
    cv2.imshow("OutputVDO", img)
    q = cv2.waitKey(1) & 0xFF
    if q == ord('c'):
        blinkCounter = 0
else:
    if ratioAvg < 35 and counter == 0:
        blinkCounter += 1
        color = (0,200,0)
        counter = 1
    elif blinkCounter > 0:
        blinkCounter = 0


cap.release()
cv2.destroyAllWindows()


