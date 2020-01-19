
import dlib

class ObjectTracker(object):
    def __init__(self):
        self.tracker = dlib.correlation_tracker()
    
    def track(self, rgbFrame, startX, startY, endX, endY):
        # construct a dlib rectangle object from the bounding
        # box coordinates and then start the dlib correlation
        # tracker
        rect = dlib.rectangle(startX, startY, endX, endY)
        self.tracker.start_track(rgbFrame, rect)

    def update(self, newRGBFrame):
        # update the tracker and grab the updated position
        self.tracker.update(newRGBFrame)

    def get_position(self):
        return self.tracker.get_position()