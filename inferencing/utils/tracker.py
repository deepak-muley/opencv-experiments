
import dlib
import cv2

class DLibObjectTracker(object):
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

class CV2MultiObjectTracker(object):
    def __init__(self):
        self.multiTracker = cv2.MultiTracker_create()

        self.tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.tracker_type = "CSRT"       
    
    def track(self, rgbFrame, startX, startY, endX, endY):

        if self.tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if self.tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if self.tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if self.tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if self.tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if self.tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if self.tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
                    
        rect = (startX, startY, endX, endY)
        return self.multiTracker.add(tracker, rgbFrame, rect)

    def update(self, newRGBFrame):
        # update the tracker and grab the updated position
        return self.multiTracker.update(newRGBFrame)

    def get_objects(self):
        return self.multiTracker.getObjects()()