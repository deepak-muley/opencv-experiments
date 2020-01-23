import cv2
from utils.frame import FrameNDArray

class VideoReader(object):
    def __init__(self, source):
        self.vcap = cv2.VideoCapture(source)

    def get_frame(self):
        if not self.vcap:
            return None
        ret, frame = self.vcap.read()
        if ret == False:
            return None
        if frame is None:
            return None
        return FrameNDArray(frame)

    def release(self):
        if self.vcap:
            self.vcap.release()

class VideoWriter(object):
    def __init__(self, dest, dim):
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        self.writer = cv2.VideoWriter(dest, fourcc, 30, dim, True)

    def write(self, frame):
        self.writer.write(frame)