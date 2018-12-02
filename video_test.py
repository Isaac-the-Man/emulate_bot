# import essentials
import cv2
import numpy as np
from emulator import process
import time

# Define hyperparameters here
VID_WIDTH = 640
VID_HEIGHT = 480
VID_FPS = 30


def main():
    # setup camera
    vid = cv2.VideoCapture(0)       # set the video mode
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, VID_WIDTH)       # decide the size of the live stream
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, VID_HEIGHT)
    vid.set(cv2.CAP_PROP_FPS, VID_FPS)        # framerate

    while True:
        print('')
        strt_t = time.time()
        _, frame = vid.read()      # read out the frame
        output_bundle = process(frame)

        # show the output
        cv2.imshow('output', output_bundle[-1])
        if cv2.waitKey(1) & 0xFF is ord('q'):       # waiting for the user to quit
            break
        end_t = time.time()
        print('time per frame: {}'.format(end_t - strt_t))

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
