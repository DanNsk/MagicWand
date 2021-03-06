# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
from threading import Event
import cv2

class PiVideoStream:
    def __init__(self, resolution=(640, 480), framerate=32, vflip = False, hflip = False, func = None):
        self.event = Event()

        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.hflip = hflip
        self.camera.vflip = vflip
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=self.camera.resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)
        self.t = None
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None

        self.func = func

    def start(self):
        # start the thread to read frames from the video stream
        self.stop()
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()

        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            frame = f.array
            if (not(self.func is None)):
                frame = self.func(frame)

            self.frame = frame

            self.rawCapture.truncate(0)

            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if (self.event.is_set()):
                self.event.clear()
                return

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.event.set()
        if (not(self.t is None)):
            self.t.join(1000)

            self.stream.close()
            self.rawCapture.close()
            self.camera.close()
        self.event.clear()
        
