import os
import time
import json

import cv2
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

from object_detection import processFrameOpenCVYoloV3, processFrameOpenCVSSDMobilenet
from human_pose import processFrameOpenCVOpenPose

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        ret, frame = cap.read()
        log.debug('Ret: {}'.format(ret))
        
        if frame is None:
            log.error('Image is none')
            break

        # Actual detection.
        #processFrameOpenCVYoloV3(frame)
        #processFrameOpenCVSSDMobilenet(frame)
        processFrameOpenCVOpenPose(frame)

        # check for 'q' key-press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            # if 'q' key-pressed break out
            break

    # close output window
    cv2.destroyAllWindows()