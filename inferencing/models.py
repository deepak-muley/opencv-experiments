from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from collections import defaultdict

class MLObjectDetectionModel(object):
    def __init__(self, modelWeightsFilePath=None, modelConfigPath=None, classLabelFilePath=None):
        self.net = None
        self.classes = []
        self.modelWeightsFilePath = modelWeightsFilePath
        self.modelConfigPath = modelConfigPath
        self.classLabelFilePath = classLabelFilePath

    @classmethod
    def Load(cls, modelWeightsFilePath=None, modelConfigPath=None, classLabelFilePath=None):
        model = MLObjectDetectionModel(modelWeightsFilePath, modelConfigPath, classLabelFilePath)
        model.load_model()
        return model

    def load_model(self):
        pass

    def detect(self):
        pass

    def get_detected_object_class_label(self, class_id):
        pass

class MLObjectDetectionModelTF(MLObjectDetectionModel):
    @classmethod
    def Load(cls):
        modelWeightsFilePath = "models/ssd_mobilenet/tf/frozen_inference_graph.pb"
        modelConfigPath = "models/ssd_mobilenet/tf/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
        classLabelFilePath = None
        model = MLObjectDetectionModelTF(modelWeightsFilePath, modelConfigPath, classLabelFilePath)
        model.load_model()
        return model

    def load_model(self):
        # Load SSD Mobilenet
        self.net = cv2.dnn.readNetFromTensorflow(self.modelWeightsFilePath, self.modelConfigPath)

        # initialize the list of class labels MobileNet SSD was trained to
        # detect coco_classes.txt
        if self.classLabelFilePath:
                with open(self.classLabelFilePath, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
        else:
            self.classes = ["background", "person", "bicycle", "car", "motorcycle",
                "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
                "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
                "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
                "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
                "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    def get_detected_object_class_label(self, class_id):
        return self.classes[class_id]

    def detect(self, frame, frameWidth, frameHeight):
        # Detecting objects
        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, size=(frameWidth, frameHeight), swapRB=True, crop=False)
        if not self.net:
            raise Exception("model is not initialized")
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections   

class MLObjectDetectionModelCaffe(MLObjectDetectionModel):
    @classmethod
    def Load(cls):
        modelWeightsFilePath = "models/ssd_mobilenet/MobileNetSSD_deploy.caffemodel"
        modelConfigPath = "models/ssd_mobilenet/MobileNetSSD_deploy.prototxt"
        classLabelFilePath = None
        model = MLObjectDetectionModelCaffe(modelWeightsFilePath, modelConfigPath, classLabelFilePath)
        model.load_model()
        return model    
    
    def load_model(self):
        # Load SSD Mobilenet
        self.net = cv2.dnn.readNetFromCaffe(self.modelConfigPath, self.modelWeightsFilePath)

        # initialize the list of class labels MobileNet SSD was trained to
        # detect
        if self.classLabelFilePath:
            with open(self.classLabelFilePath, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"]

    def get_detected_object_class_label(self, class_id):
        return self.classes[class_id]

    def detect(self, frame, frameWidth, frameHeight):
        # Detecting objects
        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (frameWidth, frameHeight), 127.5)
        if not self.net:
            raise Exception("model is not initialized")
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections
    
class MLObjectDetectionModelYolov3(MLObjectDetectionModel):
    @classmethod
    def Load(cls):
        modelWeightsFilePath = "models/yolov3/yolov3.weights"
        modelConfigPath = "models/yolov3/yolov3.cfg"
        classLabelFilePath = "models/yolov3/coco.names"
        model = MLObjectDetectionModelYolov3(modelWeightsFilePath, modelConfigPath, classLabelFilePath)
        model.load_model()
        return model
        
    def load_model(self):
        # Load Yolo
        self.net = cv2.dnn.readNet(self.modelWeightsFilePath, self.modelConfigPath)
        with open(self.classLabelFilePath, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def get_detected_object_class_label(self, class_id):
        return self.classes[class_id]

    def detect(self, frame, frameWidth, frameHeight):
        # set the status and initialize our new set of object trackers
        if not self.net:
            raise Exception("model is not initialized")
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (frameWidth, frameHeight), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        detections = self.net.forward(output_layers)
        return detections