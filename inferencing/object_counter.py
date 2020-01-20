# USAGE
# To read and write back out to video:
# python object_counter.py --prototxt models/mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#      --model models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#      --output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python object_counter.py --prototxt models/mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#      --model models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#      --output output/webcam_output.avi


# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import numpy as np
from collections import defaultdict

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from pyimagesearch.fps import FPS

from object_detection_models import *
from tracker import *
from camera import *

skip_frames = 30

def detectNewObjectsSSDMobileNetTF(frame, frameWidth, frameHeight, model, rgb, countMap, track_classes):

    detections = model.detect(frame, frameWidth, frameHeight)

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

            tracker = ObjectTracker()
            tracker.track(rgb, left, top, right, bottom)

            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            newTrackers.append(tracker)
    return newTrackers

def detectNewObjectsSSDMobileNetCaffe(frame, frameWidth, frameHeight, model, rgb, countMap, track_classes):

    detections = model.detect(frame, frameWidth, frameHeight)

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

            tracker = ObjectTracker()
            tracker.track(rgb, startX, startY, endX, endY)

            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            newTrackers.append(tracker)
    return newTrackers

def detectNewObjectsYolov3(frame, frameWidth, frameHeight, model, rgb, countMap, track_classes):
    font = cv2.FONT_HERSHEY_PLAIN
    height, width, channels = frame.shape

    outs = model.detect(frame, 416, 416)

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
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

            tracker = ObjectTracker()
            tracker.track(rgb, startX, startY, endX, endY)

            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            newTrackers.append(tracker)
    return newTrackers

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    ap.add_argument("-m", "--model", type=str, default="yolov3",
        help="which model to use for detection: ssd_caffe, ssd_tf, yolov3")        

    args = vars(ap.parse_args())

    if args["model"] == "yolov3":
        model = MLObjectDetectionModelYolov3.Load()
    elif args["model"] == "ssd_caffe":
        model = MLObjectDetectionModelCaffe.Load()
    elif args["model"] == "ssd_tf":
        model = MLObjectDetectionModelTF.Load()

    # if a video path was not supplied, grab a reference to the webcam
    if not args.get("input", False):
        print("[INFO] starting video stream...")
        vs = VideoReader(0)
    # otherwise, grab a reference to the video file
    else:
        print("[INFO] opening video file...")
        vs = VideoReader(args["input"])

    # initialize the video writer (we'll instantiate later if need be)
    writer = None

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    frameWidth = None
    frameHeight = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}
    countMap = defaultdict(lambda:0)

    # start the frames per second throughput estimator
    fps = FPS().start()

    totalFrames = 0

    # loop over frames from the video stream
    while True:
        # grab the next frame and handle
        frame = vs.get_frame()

        if frame is None:
            break

        # resize the frame,
        # then convert the frame from BGR to RGB for dlib
        scale_percent = 60 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image        
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if frameWidth is None or frameHeight is None:
            (frameHeight, frameWidth) = frame.shape[:2]

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args["output"] is not None and writer is None:
            writer = VideoWriter(args["output"], (frameWidth, frameHeight))

        # initialize the list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        track_classes = [ "person", "bicycle", "car" ]
        if totalFrames % skip_frames == 0:
            if args["model"] == "yolov3":
                trackers = detectNewObjectsYolov3(frame, frameWidth, frameHeight, model, rgb, countMap, track_classes)
            elif args["model"] == "ssd_caffe":
                trackers = detectNewObjectsSSDMobileNetCaffe(frame, frameWidth, frameHeight, model, rgb, countMap, track_classes)
            elif args["model"] == "ssd_tf":
                trackers = detectNewObjectsSSDMobileNetTF(frame, frameWidth, frameHeight, model, rgb, countMap, track_classes)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        #construct a tuple of information we will be displaying on the frame
        info = [
        	("Total People Seen", countMap["person"]),
        	("Total Bycycles Seen", countMap["bicycle"]),		
        	("Total Cars Seen", countMap["car"])
        ]

        # loop over the info tuples and draw them on our frame
        # for (i, (k, v)) in enumerate(info):
        # 	text = "{}: {}".format(k, v)
        # 	cv2.putText(frame, text, (10, frameHeight - ((i * 20) + 20)),
        # 		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    vs.release()

    # close any open windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()