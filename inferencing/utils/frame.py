import numpy as np
import argparse
import time
import cv2
import numpy as np
from collections import defaultdict

from utils.object_detection_models import *
from utils.tracker import DLibObjectTracker, CV2MultiObjectTracker

class FrameNDArray(object):
    def __init__(self, frame):
        self.frameNumpyNDArray = frame
        self.rgbFrameNDArray = None

    @property
    def frame(self):
        return self.frameNumpyNDArray

    @property
    def rgbFrame(self):
        if self.rgbFrameNDArray is None:
            self.rgbFrameNDArray = FrameNDArray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        return self.rgbFrameNDArray.frame

    @property
    def frame_shape(self):
        return self.frame.shape      

    def resizeToScalePercent(self, scale_percent):
        width = int(self.frame_shape[1] * scale_percent / 100)
        height = int(self.frame_shape[0] * scale_percent / 100)
        dim = (width, height)
        return FrameNDArray(cv2.resize(self.frame, dim, interpolation=cv2.INTER_AREA))

    def detectNewObjectsSSDMobileNetTF(self, frameWidth, frameHeight, model, countMap, track_classes):
        
        detections = model.detect(self.frame, frameWidth, frameHeight)

        # Showing informations on the screen
        newTrackers = []

        # Loop on the outputs
        for detection in detections[0,0,:,:]:
            score = float(detection[2])
            if score > 0.2:
                # extract the index of the class label from the
                # detections list
                idx = int(detection[1])  # prediction class index.
                label = model.get_detected_object_class_label(idx)

                # if the class label is not a person, ignore it
                if label not in track_classes:
                    continue

                countMap[label] += 1

                left = int(detection[3] * frameWidth)
                top = int(detection[4] * frameHeight)
                right = int(detection[5] * frameWidth)
                bottom = int(detection[6] * frameHeight)

                tracker = DLibObjectTracker()
                tracker.track(self.rgbFrame, left, top, right, bottom)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                newTrackers.append(tracker)
        return newTrackers

    def detectNewObjectsSSDMobileNetCaffe(self, frameWidth, frameHeight, model, countMap, track_classes):

        detections = model.detect(self.frame, frameWidth, frameHeight)

        # Showing informations on the screen
        newTrackers = []

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            if confidence > 0.1:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])
                label = model.get_detected_object_class_label(idx)

                # if the class label is not a person, ignore it
                if label not in track_classes:
                    continue

                countMap[label] += 1

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = DLibObjectTracker()
                tracker.track(self.rgbFrame, startX, startY, endX, endY)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                newTrackers.append(tracker)
        return newTrackers

    def detectNewObjectsYolov3(self, frameWidth, frameHeight, model, countMap, track_classes):
        font = cv2.FONT_HERSHEY_PLAIN
        height, width, channels = self.frame_shape

        outs = model.detect(self.frame, 416, 416)

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.2
        nms_threshold = 0.4

        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        newTrackers = []
        totalPeopleSeen = 0
        totalBicyclesSeen = 0
        totalCarsSeen = 0    

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = model.get_detected_object_class_label(class_ids[i])
                confidence = confidences[i]

                # if the class label is not a person, ignore it
                if label not in track_classes:
                    continue

                countMap[label] += 1

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                (startX, startY, endX, endY) = (x, y, x + w, y + h)

                tracker = DLibObjectTracker()
                tracker.track(self.rgbFrame, startX, startY, endX, endY)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                newTrackers.append(tracker)
        return newTrackers