

from threading import Thread
import numpy as np
import cv2

small = 40
medium = 120
large = 240

class backgroundCalculator:

    def __init__(self, firstThirtySumB,firstThirtySumG,firstThirtySumR,previousTwentySumB,previousTwentySumG,previousTwentySumR,lastTenSumB,lastTenSumG,lastTenSumR,firstThirty,previousTwenty,lastTenFrames, backgroundAveB,backgroundAveG,backgroundAveR, frame, count = 1):

        self.firstThirty = firstThirty
        self.previousTwenty = previousTwenty
        self.lastTenFrames = lastTenFrames

        self.previousTwentySumB = previousTwentySumB
        self.previousTwentySumG = previousTwentySumG
        self.previousTwentySumR = previousTwentySumR

        self.firstThirtySumB = firstThirtySumB
        self.firstThirtySumG = firstThirtySumG
        self.firstThirtySumR = firstThirtySumR

        self.lastTenSumB = lastTenSumB
        self.lastTenSumG = lastTenSumG
        self.lastTenSumR = lastTenSumR

        self.backgroundAveB = backgroundAveB
        self.backgroundAveG = backgroundAveG
        self.backgroundAveR = backgroundAveR

        self.count = count
        self.newFrameAvailable = False
        self.newFrame = frame


    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            while not self.newFrameAvailable:
                stall = 0
            B, G, R = cv2.split(self.newFrame)

            # frame[frame >= 240] = 0
            # frame = cv2.equalizeHist(frame)


            b, g, r  = cv2.split(self.lastTenFrames[self.count % small])
            b2,g2,r2 = cv2.split(self.previousTwenty[self.count % medium])
            b3,g3,r3 = cv2.split(self.firstThirty[self.count % large])

            self.lastTenSumB = self.lastTenSumB + B - b
            self.lastTenSumG = self.lastTenSumB + G - g
            self.lastTenSumR = self.lastTenSumB + R - r

            self.previousTwentySumB = self.previousTwentySumB + b - b2
            self.previousTwentySumG = self.previousTwentySumG + g - g2
            self.previousTwentySumR = self.previousTwentySumR + r - r2

            self.firstThirtySumB = self.firstThirtySumB + b2 - b3
            self.firstThirtySumG = self.firstThirtySumG + g2 - g3
            self.firstThirtySumR = self.firstThirtySumR + r2 - r3


            self.firstThirty[self.count%large] = self.previousTwenty[self.count % medium]
            self.previousTwenty[self.count % medium] = self.lastTenFrames[self.count % small]
            self.lastTenFrames[self.count % small] = self.newFrame

            self.backgroundAveB = self.lastTenSumB * .0075 + self.previousTwentySumB * .0033 + self.firstThirtySumB * .00125
            self.backgroundAveG = self.lastTenSumG * .0075 + self.previousTwentySumG * .0033 + self.firstThirtySumG * .00125
            self.backgroundAveR = self.lastTenSumR * .0075 + self.previousTwentySumR * .0033 + self.firstThirtySumR * .00125

            self.newFrameAvailable = False

    def getbackgroundAve(self):
        return cv2.merge((self.backgroundAveB, self.backgroundAveG, self.backgroundAveR))


    def addNewFrame(self, frame, backgroundAve, mask):
        aveFrame = mask & 0
        aveFrame[mask == 255] = 0
        aveFrame[mask == 0] = 255
        aveFrame = cv2.merge((aveFrame, aveFrame, aveFrame))
        aveFrame = aveFrame & frame
        backgroundAve = backgroundAve.astype(np.uint8)
        mask = cv2.merge((mask, mask, mask))
        temp = mask & backgroundAve
        aveFrame = aveFrame | temp
        self.newFrame = aveFrame.astype(np.float32)
        self.count = self.count + 1
        self.newFrameAvailable = True
