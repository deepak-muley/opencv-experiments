#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
#http://www.life2coding.com/load-webcam-video-using-opencv-python/
#https://www.life2coding.com/save-webcam-video-feed-file-using-opencv-python/

import cv2
 
def save_webcam(outPath,fps,mirror=False):
    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)
 
    currentFrame = 0
 
    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
 
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outPath, fourcc, fps, (int(width), int(height)))
 
    while (cap.isOpened()):
 
        # Capture frame-by-frame
        ret, frame = cap.read()
 
        if ret == True:
            if mirror == True:
                # Mirror the output video frame
                frame = cv2.flip(frame, 1)
            # Saves for video
            out.write(frame)
 
            # Display the resulting frame
            cv2.imshow('frame', frame)
        else:
            break
 
        if cv2.waitKey(1) & 0xFF == ord('q'):  # if 'q' is pressed then quit
            break
 
        # To stop duplicate images
        currentFrame += 1
 
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
 
def main():
    save_webcam('output.avi', 30.0,mirror=True)
 
if __name__ == '__main__':
    main()