# import essential package
import cv2
import numpy as np
import argparse

# Define constant
ROI_X_OFFSET = 1/10
ROI_Y_OFFSET = 1/3
GAUSSIANBLUR_KERNEL = (7,7)
HSV_MIN = (47,87,44)
HSV_MAX = (101,255,255)
MIN_TILE_AREA = 200
MAX_TILE_COUNT = 4
MIN_LINE_AREA = 150
THRESHED_LOW = 35
THRESHED_HIGH = 255
SPLIT_PORTION = 1/9
MAX_TURN = np.pi/2

# get the arguments for frame to frame test
def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True, help = 'enter the image path')
    ap.add_argument('-s', '--scale', required = True, help = 'scale factor of output')
    args = vars(ap.parse_args())        # take in the arguments
    return args

# get the width and height of an image
def get_geometry(img):
    height, width = img.shape[:2]
    return height, width

# get the green tiles
def locate_green(hsv_img):
    green_img = cv2.inRange(hsv_img, HSV_MIN, HSV_MAX)        # mask out the green tiles
    green_ROI_img = get_ROI(green_img)
    _, contours, _ = cv2.findContours(green_ROI_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('Total contours found: {}'.format(len(contours)))

    # sort area from large to small, only MAX_TILE_COUNT of tiles is allowed
    contours.sort(key=cv2.contourArea, reverse=True)
    contours = contours[:MAX_TILE_COUNT]        # only top areas are left

    # filter out contours that has too small area
    cnts_c = []     # contour centers
    filtered_cnts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= MIN_TILE_AREA:
            filtered_cnts.append(cnt)
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cnts_c.append((cx,cy))
    print('Real contours found: {}'.format(len(contours)))
    return filtered_cnts, cnts_c

# get the ROI, which is a trapezoid
def get_ROI(img):
    mask = np.zeros_like(img)
    ignore_mask_color = 255

    height, width = get_geometry(img)
    vertices = np.array([[0,height],[ROI_X_OFFSET*width,ROI_Y_OFFSET*height],[(1-ROI_X_OFFSET)*width, ROI_Y_OFFSET*height],[width, height]], np.int32)
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# slice and get midpoint
def scan_slice(img):
    height, width = get_geometry(img)
    slice_h = int(height*(1-SPLIT_PORTION))
    n_slice_img = img[:slice_h,:]
    slice_img = img[slice_h:,:]

    # find contour and midpoint
    _, contours, _ = cv2.findContours(slice_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        print('line contours found: {}'.format(len(contours)))
        M = cv2.moments(contours[0])
        area = cv2.contourArea(contours[0])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00']) + slice_h
        return n_slice_img, slice_img, [(cx, cy), area ]
    else:
        print('no line contours found')
        return n_slice_img, slice_img, None

# algorithm to calculate direction with midpoint
def get_direction(img, midpoint):
    cx, cy = midpoint[0]
    area = midpoint[1]
    height, width = get_geometry(img)

    # the activation function for shift
    def activate_shift(shift):
        shift_p = np.round(shift/(width/2),3)        # make shift a percentage
        print('Shift percentage: {}'.format(shift_p))
        func = lambda x : x        # define the function
        return MAX_TURN*np.round(func(shift_p),3)

    # calculate the turn
    shift = int(cx - width/2)
    print('Pixel shift: {}'.format(shift))
    turn = activate_shift(shift)
    return turn

def process(input_img):
    # copy of input_img for drawing, get heith and width
    output_img = input_img.copy()
    height, width = get_geometry(output_img)

    # reduce noise
    blurred_img = cv2.GaussianBlur(input_img, GAUSSIANBLUR_KERNEL, 0)

    # get hsv channel
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    # locate the green tiles, and draw the center
    cnts, cnts_c = locate_green(hsv_img)
    if cnts_c and cnts is not None:      # filter out the case where there is not green tiles found
        cv2.drawContours(output_img, cnts, -1, (0,255,0), 3)
        for cx,cy in cnts_c:
            cv2.circle(output_img, (cx,cy), 5, (255,0,0), -1)

    # grey, threshold, and get ROI
    grey_img = cv2.cvtColor(blurred_img.copy(), cv2.COLOR_BGR2GRAY)
    _,threshed_img = cv2.threshold(grey_img.copy(),THRESHED_LOW,THRESHED_HIGH,cv2.THRESH_BINARY_INV)
    ROI_img = get_ROI(threshed_img.copy())

    # get line midpoint using contours
    _, slice_img, midpoint = scan_slice(ROI_img)
    if midpoint is not None:       # filters out the case for no midline found
        cv2.circle(output_img, midpoint[0], 5, (0,0,255), -1)

        # get the direction
        turn = get_direction(ROI_img, midpoint)
        print('The turn is {}'.format(turn*180/np.pi))
        cv2.line(output_img, (int(width/2), 0), (int(width/2), height), (255,0,0), 3)

    img_bundle = [blurred_img, hsv_img, grey_img, threshed_img, ROI_img, slice_img, output_img]
    return img_bundle

def main():
    # read the arguments
    args = get_arguments()
    path = args['image']
    scale = int(args['scale'])

    # read and scale the image
    raw_img = cv2.imread(path)
    raw_height, raw_width = get_geometry(raw_img)
    scaled_height = int(raw_height/scale)
    scaled_width = int(raw_width/scale)
    resized_img = cv2.resize(raw_img, (scaled_width, scaled_height))

    # process
    output_bundle = process(resized_img)

    # show the frames
    names = 'blurred_img,hsv_img,grey_img,threshed_img,ROI_img,slice_img,output_img'.split(',')
    counter = 0
    for img in output_bundle:
        cv2.imshow(names[counter], img)
        counter += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
