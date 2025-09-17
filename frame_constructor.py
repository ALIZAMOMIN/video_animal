# The following two methods are taken from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
import cv2
import numpy as np
IMG_SIZE = 224

'''
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

'''

#converting to frames
def frame_cons(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    #path of video file
    #0>>read all the frames
    #size 224X224 

    cap = cv2.VideoCapture(path) #load  video
    frames = []
    try:
        while True:
            ret, frame = cap.read()#ret >cheks if the frame is read successully  #frame >nparray
            #cap.read()    the next frame from the video
            #returns 2 values
            if not ret:
                break
            #frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]] #from bgr to rgb 
            frames.append(frame)  

            if len(frames) == max_frames:
                break
    finally:
        cap.release() #release system resource
    return np.array(frames)

#frame shape 224 ,224,3 <<color channel
