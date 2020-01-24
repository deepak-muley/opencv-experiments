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

from utils.object_detection_models import *
from utils.tracker import DLibObjectTracker, CV2MultiObjectTracker
from utils.camera import VideoReader, VideoWriter

skip_frames = 30



def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    ap.add_argument("-m", "--model", type=str, default="yolov3",
        help="which model to use for detection: ssd_caffe, ssd_tf, yolov3")
    ap.add_argument("-t", "--tracker", type=str, default="cv2",
        help="which tracker to use: cv2 or dlib")               

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
    totalDown = 0
    totalUp = 0    

    # loop over frames from the video stream
    while True:
        # grab the next frame and handle
        frameNDArray = vs.get_frame()

        if frameNDArray is None:
            break

        # resize the frame,
        # then convert the frame from BGR to RGB for dlib
        scale_percent = 60 # percent of original size
        # resize image        
        frameNDArray = frameNDArray.resizeToScalePercent(scale_percent)
        rgbFrame = frameNDArray.rgbFrame

        # if the frame dimensions are empty, set them
        if frameWidth is None or frameHeight is None:
            (frameHeight, frameWidth) = frameNDArray.frame_height_width

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args["output"] is not None and writer is None:
            writer = VideoWriter(args["output"], (frameWidth, frameHeight))

        # initialize the list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        rects = []
        tracker_type = args["tracker"]

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        track_classes = [ "person" ]
        if totalFrames % skip_frames == 0:
            if args["model"] == "yolov3":
                trackers = frameNDArray.detectNewObjectsYolov3(frameWidth, frameHeight, model, countMap, track_classes)
            elif args["model"] == "ssd_caffe":
                trackers = frameNDArray.detectNewObjectsSSDMobileNetCaffe(frameWidth, frameHeight, model, countMap, track_classes)
            elif args["model"] == "ssd_tf":
                trackers = frameNDArray.detectNewObjectsSSDMobileNetTF(frameWidth, frameHeight, model, countMap, track_classes)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # update the tracker and grab the updated position
                tracker.update(rgbFrame)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        frameNDArray.drawHorizontalLineInCenter()
        
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

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < frameHeight // 2:
                        totalUp += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > frameWidth // 2:
                        totalDown += 1
                        to.counted = True                

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frameNDArray.frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frameNDArray.frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        #construct a tuple of information we will be displaying on the frame
        info = [
            ("Up", totalUp),
            ("Down", totalDown)           
        ]

        #loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
        	text = "{}: {}".format(k, v)
        	cv2.putText(frameNDArray.frame, text, (10, frameHeight - ((i * 20) + 20)),
        		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frameNDArray.frame)

        # show the output frame
        cv2.imshow("Frame", frameNDArray.frame)
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