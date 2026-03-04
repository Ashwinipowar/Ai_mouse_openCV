import cv2
import numpy as np
import time
import autopy
import handtracking as htm


wCam, hCam = 640, 480
frameR = 80
smoothening = 5


pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

clickTime = 0
cooldown = 0.3

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        cv2.rectangle(img, (frameR, frameR),
                      (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # Move Mode
        if fingers[1] == 1 and fingers[2] == 0:

            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(wScr - clocX, clocY)

            cv2.circle(img, (x1, y1), 10,
                       (255, 0, 255), cv2.FILLED)

            plocX, plocY = clocX, clocY

        # Click Mode
        if fingers[1] == 1 and fingers[2] == 1:

            length, img, lineInfo = detector.findDistance(
                (lmList[8][1], lmList[8][2]),
                (lmList[12][1], lmList[12][2]),
                img, draw=False)

            if length < 30:
                currentTime = time.time()
                if currentTime - clickTime > cooldown:
                    clickTime = currentTime
                    autopy.mouse.click()
                    cv2.circle(img,
                               (lineInfo[4], lineInfo[5]),
                               12,
                               (0, 255, 0),
                               cv2.FILLED)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()