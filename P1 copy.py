import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv, numpy as np


# test image

# img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
# img_line = cv.line(img,(0,0),(511,511),(255,0,0),5)

# read input images
image = mpimg.imread("test_images/solidWhiteCurve.jpg")
# image = cv.imread("test_images/solidWhiteCurve.jpg")


# Define a function to processing images inside the directory
def image_processing(image):
    
    """ This functions'output will be the processed images and as input raw images
    from test_images/ directory """
    
    # gray scale
    gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # gaussian blur
    kernel = (3, 3)
    sigma_x = 0
    gaussian_blur = cv.GaussianBlur(src = image, ksize= kernel, sigmaX = sigma_x)
    # define low & hight threshold values for canny()
    low_threshold, high_threshold = 10, 30
    # apply canny edge detector function
    canny = cv.Canny(image, low_threshold, high_threshold)

    return canny

# Find vertices

# define a blank mask to start with
# define vertices [[x0,y0],[x1, y1],[x2,y2],[x3, y3]]

# Anti clockwise points 
# x0, y0 --> x1, y1 (left line)

# image.shape --> (540, 960, 3)

# x2, y2 --> x3, y3 (right line)
image_height = image.shape[0]
vertical_line = image_height/2

image_width = image.shape[1]
horizontal_line = image_width/2

# Apex is defined as the middle point of vertical and horizontal line intersection
apex = [vertical_line, horizontal_line]
# left_vert = apex, 540
# right_vert = 960,apex
# left_diagn_line = apex, [0, 540]
# right_diagn_line = apex, [960, 540]
# image_vertices = left_diagn_line, right_diagn_line
# vertices = np.array([image_vertices], np.int32)
# print('new vertices', left_diagn_line, right_diagn_line)

# image vertices are chosen based on a triangular area
# print('image vertices', image_vertices)

# coordiates [x0, y0], [x1, y1]
xy_top_left, xy_top_right = [460, 330], [510, 330]
xy_bottom_left, xy_bottom_right = [310, 450], [700, 450]

vertices = np.array([xy_top_left, xy_bottom_left , xy_bottom_right, xy_top_right], np.int32)
# changing the coordinate order will affect the selected region of interest
# vertices = np.array([ xy_bottom_left, xy_top_left, xy_bottom_right, xy_top_right], np.int32)


def region_of_interest(image, vertices):
	# define a blank mask
    mask = np.zeros_like(image)   
    if len(image.shape) > 2:
        # i.e. 3 or 4 depending on your image
        channel_count = image.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon defined by "vertices" with the fill color    
    fill_poly = cv.fillPoly(img = mask, pts = [vertices], 
    color = ignore_mask_color)
    # Returning the image only where mask pixels are nonzero
    masked_image = cv.bitwise_and(src1 = image, src2 = mask)
    return masked_image
# plt.imshow(region_of_interest(image, vertices))


def draw_lines(image, lines, color=[255, 0, 0], thickness=10):

    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.	
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # Avoid drawing lines if error is encountered
    if lines is None or len (lines) == 0:
        return
    # Set a flag state for drawing left, right lines
    draw_right_line, draw_left_line = True, True
    
    for line in lines:
        x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]

    # plt.imshow(draw_lines(image, lines, color=[255, 0, 0], thickness=10))


# cv.imshow('images', image_processing(image))
cv.imshow('Region of interest', region_of_interest(image, vertices))
# cv.imshow('Draw lines', draw_lines(image, lines, color=[255, 0, 0], thickness=10))


cv.waitKey(0)
cv.destroyAllWindows()
