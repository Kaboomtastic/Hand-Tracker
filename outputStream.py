
import cv2
from threading import Thread
import matplotlib.pyplot as plt

class outputStream:
    def __init__(self, initframe):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.nextFrame = initframe
        self.thisFrame = initframe
        self.stopped = False
        #self.mywindow = cv2.namedWindow("name",flags= cv2.WINDOW_AUTOSIZE)
        self.this = True



    def start(self):
        # start the thread to read frames from the video stream

        Thread(target=self.update, args=()).start()
        plt.imshow(self.thisFrame)
        print("ahhh")
        return self

    def update(self):
        idk = 0
        # keep looping infinitely until the thread is stopped

        #while self.this == True:
        # if the thread indicator variable is set, stop the thread
        #if self.stopped:
        #    return
        #print("working")
        # otherwise, read the next frame from the stream
        # Display the resulting frame

        #cv2.waitKey(1)
        #self.thisFrame = self.nextFrame
        #self.this = False

    def write(self, nextFrame):
        # return the frame most recently read
        self.nextFrame = nextFrame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
