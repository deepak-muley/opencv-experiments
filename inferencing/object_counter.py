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
from pyimagesearch.centroidtracker import CentroidTracker
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
from models import *

skip_frames = 30

def load_yolov3_model():
    # Load Yolo
    return MLObjectDetectionModelYolov3.Load()

def load_ssd_mobilenet_model_caffe():
    # Load SSD Mobilenet
    return MLObjectDetectionModelCaffe.Load()

def load_ssd_mobilenet_model_tf():
    return MLObjectDetectionModelTF.Load()

def detectNewObjectsSSDMobileNetTF(frame, frameWidth, frameHeight, model, rgb, countMap):

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
            if label not in [ "person", "bicycle", "car" ]:
                continue

            countMap[label] += 1

            left = int(detection[3] * frameWidth)
            top = int(detection[4] * frameHeight)
            right = int(detection[5] * frameWidth)
            bottom = int(detection[6] * frameHeight)

            (startX, startY, endX, endY) = (left, top, right, bottom)

            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)

            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            newTrackers.append(tracker)
    return newTrackers

def detectNewObjectsSSDMobileNetCaffe(frame, frameWidth, frameHeight, model, rgb, countMap):

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
            if label not in [ "person", "bicycle", "car" ]:
                continue

            countMap[label] += 1

            # compute the (x, y)-coordinates of the bounding box
            # for the object
            box = detections[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
            (startX, startY, endX, endY) = box.astype("int")

            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)

            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            newTrackers.append(tracker)
    return newTrackers

def detectNewObjectsYolov3(frame, frameWidth, frameHeight, model, rgb, countMap):
    font = cv2.FONT_HERSHEY_PLAIN
    height, width, channels = frame.shape

    outs = model.detect(frame, 416, 416)

    class_ids = []
    confidences = []
    boxes = []

    # loop over the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

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
            if label not in [ "person", "bicycle", "car" ]:
                continue

            countMap[label] += 1

            # compute the (x, y)-coordinates of the bounding box
            # for the object
            (startX, startY, endX, endY) = (x, y, x + w, y + h)

            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)

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
        model = load_yolov3_model()
    elif args["model"] == "ssd_caffe":
        model = load_ssd_mobilenet_model_caffe()
    elif args["model"] == "ssd_tf":
        model = load_ssd_mobilenet_model_tf()

    # if a video path was not supplied, grab a reference to the webcam
    if not args.get("input", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
    # otherwise, grab a reference to the video file
    else:
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(args["input"])

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
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if args["input"] is not None and frame is None:
            break

        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if frameWidth is None or frameHeight is None:
            (frameHeight, frameWidth) = frame.shape[:2]

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (frameWidth, frameHeight), True)

        # initialize the list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % skip_frames == 0:
            if args["model"] == "yolov3":
                trackers = detectNewObjectsYolov3(frame, frameWidth, frameHeight, model, rgb, countMap)
            elif args["model"] == "ssd_caffe":
                trackers = detectNewObjectsSSDMobileNetCaffe(frame, frameWidth, frameHeight, model, rgb, countMap)
            elif args["model"] == "ssd_tf":
                trackers = detectNewObjectsSSDMobileNetTF(frame, frameWidth, frameHeight, model, rgb, countMap)

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

        # # loop over the info tuples and draw them on our frame
        # for (i, (k, v)) in enumerate(info):
        # 	text = "{}: {}".format(k, v)
        # 	cv2.putText(frame, text, (10, frameHeight - ((i * 20) + 20)),
        # 		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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

    # if we are not using a video file, stop the camera video stream
    if not args.get("input", False):
        vs.stop()
    # otherwise, release the video file pointer
    else:
        vs.release()

    # close any open windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()