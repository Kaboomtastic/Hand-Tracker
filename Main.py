from __future__ import print_function
from threading import Thread
import numpy as np
import cv2
from webcamStream import webcamStream
from imutils.video import FPS
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from outputStream import outputStream
from IPython import display
import PIL
import IPython
from io import BytesIO
from IPython.display import clear_output, Image, display
from backgroundCalculator import backgroundCalculator
from skimage.metrics import structural_similarity as ssim
from regionGenerator import generateProposals
import timeit
import multiprocessing
from multiprocessing import Process


def main(reader):

    small = 40
    medium = 120
    large = 240

    previousThresh = 0


    lastTenFrames = []
    previousTwenty = []
    firstThirty = []
    count = 1

    frame = reader.recv()
    frame,e,d = cv2.split(frame)

    lastTenSumB = frame * 0
    lastTenSumB = lastTenSumB.astype(np.int32)
    lastTenSumG = lastTenSumB
    lastTenSumR = lastTenSumB

    previousTwentySumB = lastTenSumB
    previousTwentySumG = lastTenSumB
    previousTwentySumR = lastTenSumB

    firstThirtySumB = lastTenSumB
    firstThirtySumG = lastTenSumB
    firstThirtySumR = lastTenSumB


    for i in range(small):

        frame = reader.recv()
        B,G,R = cv2.split(frame)
        lastTenSumB = lastTenSumB + B
        lastTenSumG = lastTenSumG + G
        lastTenSumR = lastTenSumR + R
        lastTenFrames.append(frame)

    for j in range(medium):

        frame = reader.recv()
        B,G,R = cv2.split(frame)
        previousTwenty.append(lastTenFrames[j % small])
        b,g,r = cv2.split(lastTenFrames[j % small])
        lastTenSumB = lastTenSumB - b
        lastTenSumB = lastTenSumB + B
        lastTenSumG = lastTenSumB - g
        lastTenSumG = lastTenSumB + G
        lastTenSumR = lastTenSumB - r
        lastTenSumR = lastTenSumB + R
        previousTwentySumB = previousTwentySumB + b
        previousTwentySumG = previousTwentySumG + g
        previousTwentySumR = previousTwentySumR + r
        lastTenFrames[j % small] = frame

    for k in range(large+1):

        frame = reader.recv()
        B,G,R = cv2.split(frame)
        b, g, r = cv2.split(lastTenFrames[j % small])
        b2,g2,r2, = cv2.split(previousTwenty[j%medium])

        lastTenSumB = lastTenSumB - b
        lastTenSumB = lastTenSumB + B
        lastTenSumG = lastTenSumB - g
        lastTenSumG = lastTenSumB + G
        lastTenSumR = lastTenSumB - r
        lastTenSumR = lastTenSumB + R
        previousTwentySumB = previousTwentySumB + b
        previousTwentySumG = previousTwentySumG + g
        previousTwentySumR = previousTwentySumR + r
        previousTwentySumB = previousTwentySumB - b2
        previousTwentySumG = previousTwentySumG - g2
        previousTwentySumR = previousTwentySumR - r2
        firstThirtySumB = firstThirtySumB + b2
        firstThirtySumG = firstThirtySumG + g2
        firstThirtySumR = firstThirtySumR + r2
        firstThirty.append(previousTwenty[j%medium])
        previousTwenty[j%medium] = lastTenFrames[j%small]
        lastTenFrames[j % small] = frame

    backgroundAveB = lastTenSumB * .01 + previousTwentySumB * .0025 + firstThirtySumB * .00125
    backgroundAveG = lastTenSumG * .01 + previousTwentySumG * .0025 + firstThirtySumG * .00125
    backgroundAveR = lastTenSumR * .01 + previousTwentySumR * .0025 + firstThirtySumR * .00125


    frame = reader.recv()

    bc = backgroundCalculator(firstThirtySumB,firstThirtySumG,firstThirtySumR,previousTwentySumB,previousTwentySumG,previousTwentySumR,lastTenSumB,lastTenSumG,lastTenSumR,firstThirty,previousTwenty,lastTenFrames, backgroundAveB,backgroundAveG,backgroundAveR, frame, count).start()


    kernel = np.array([(0,1,0),(1,1,1),(0,1,0)])
    kernel = kernel.astype(np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    kernel3 = np.ones((3,3),np.uint8)


    fps = FPS().start()
    times = []
    delay = 0
    for i in range(100):

        frame = reader.recv()

        start = timeit.default_timer()
        original = frame.astype(np.uint8)

        backgroundAve = bc.getbackgroundAve()

        output = abs(frame - backgroundAve)
        #output = output.astype(np.uint8)

        diffH,diffS,diffV = cv2.split(output)
        stop = timeit.default_timer()


        totalDiff = .5*np.square(diffH) + .5*np.square(diffV) + .5*np.square(diffS)
        threshold = np.average(totalDiff) + np.std(totalDiff)

        mask = cv2.threshold(totalDiff, threshold, 255, cv2.THRESH_BINARY)[1]

        mask = mask.astype(np.uint8)
        mask2 = np.copy(mask)

        #if i % 2 == 0:
        bc.addNewFrame(original,backgroundAve,mask)

        proposals = []
        generateProposals(original,mask2,0,proposals,0,0)

        #for a in range(len(proposals)):
        #    temp = proposals[a].image
        #    temp = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
        #    filename = '/Users/Kaboomtastic/ownCloud/DIP project Training Images/OurImages/im' + str(i*50+a) + '.jpg'
        #    cv2.imwrite(filename, temp)

        #if i %4 == 0:
        #    cv2.imshow("output",cv2.cvtColor(disp, cv2.COLOR_HSV2BGR))

        #Your statements here

        times.append(stop - start)

        fps.update()

    fps.stop()

    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print(np.average(times))
    # When everything done, release the capture

    cv2.destroyAllWindows()


dim = (320, 180)

reader, writer = multiprocessing.Pipe(False)
# producer process
video_process = Process(target=main, args=[reader])
video_process.start()  # Launch capture process
    # The VideoCapture class wraps the video acquisition logic
stream = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = stream.read() # the method returns the next frame
    smaller = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    smaller = cv2.cvtColor(smaller, cv2.COLOR_BGR2HSV)
    #smaller = smaller.astype(np.float32)
    writer.send(smaller)  # send the new frame to the consumer process
