


class backngroundBackgroundCalculator:

    small = 40
    medium = 120
    large = 240


    def __init__(self):
        this = 1


    def update(self,firstThirtySumB,firstThirtySumG,firstThirtySumR,previousTwentySumB,previousTwentySumG,previousTwentySumR,lastTenSumB,lastTenSumG,lastTenSumR,firstThirty,previousTwenty,lastTenFrames, backgroundAveB,backgroundAveG,backgroundAveR, frame, count = 1):
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

        while True:
            while not self.newFrameAvailable:
                stall = 0
                #do frame receive here

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

            #frame send here
