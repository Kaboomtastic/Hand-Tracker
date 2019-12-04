# import the necessary packages
from threading import Thread
from multiprocessing import Process
import numpy as np
import cv2


class webcamStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)

        (self.grabbed, self.frame) = self.stream.read()

        self.scale_percent = 25  # percent of original size
        width = int(self.frame.shape[1] * self.scale_percent / 100)
        height = int(self.frame.shape[0] * self.scale_percent / 100)
        #dim = (width, height)
        dim = (320, 180)
        print(width)
        print(height)
        # resize image
        smaller = cv2.resize(self.frame, dim, interpolation=cv2.INTER_AREA)
        smaller = smaller.astype(np.float32)
        self.frame = smaller

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        self.updated = True

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()

        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            # otherwise, read the next frame from the stream
            if self.stopped:
                return


            (self.grabbed, frame) = self.stream.read()

            dim = (320, 180)
            # resize image
            smaller = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            smaller = cv2.cvtColor(smaller, cv2.COLOR_BGR2HSV)
            #H,S,V = cv2.split(smaller)
            #V = cv2.equalizeHist(V)
            #smaller = cv2.merge((H,S,V))

            self.frame = smaller

            self.updated = True

    def read(self):
        # return the frame most recently read
        self.updated = False
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()

    def hasUpdate(self):
        return self.updated
