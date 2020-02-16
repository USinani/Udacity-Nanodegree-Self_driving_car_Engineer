import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv, numpy as np
import os




# Global variables

# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
# define low & hight threshold values for canny()
low_threshold, high_threshold = 10, 30

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 0.85
trap_top_width = 0.07
trap_height = 0.4

# Hough Transform
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments


# define a blank mask to start with

# define vertices [[x0,y0],[x1, y1],[x2,y2],[x3, y3]]
# Anti clockwise points 
# x0, y0 --> x1, y1 (left line)
# x2, y2 --> x3, y3 (right line)
vertices = np.array([[460, 330], [310, 450], 
[700, 450], [510, 330]], np.int32)


# Test images

def test_images():
    return os.listdir('test_images/')

# read input images
# imgs = mpimg.imread("test_images/solidWhiteCurve.jpg")
# OpenCV read function
imgs = cv.imread('test_images/solidWhiteCurve.jpg')


def grayscale(imgs):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    # image = cv.imread(imgs)
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    return cv.cvtColor(imgs, cv.COLOR_BGR2GRAY)


def gaussian_blur(imgs, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv.GaussianBlur(imgs, (kernel_size, kernel_size), 0)

'''
def canny(imgs, low_threshold, high_threshold):

    """Applies the Canny transform"""   
    # gray scale
    gray_scale = cv.cvtColor(imgs, cv.COLOR_BGR2GRAY)
    # gaussian blur
    gaussian_blur = cv.GaussianBlur(src = imgs, ksize= (3, 3), sigmaX = 0)
    # apply canny edge detector function
    # canny = cv.Canny(imgs, low_threshold, high_threshold)

    return cv.Canny(imgs, low_threshold, high_threshold)
'''

def canny(imgs, low_threshold, high_threshold):
    # gray scale
    gray_scale = cv.cvtColor(imgs, cv.COLOR_BGR2GRAY)
    # gaussian blur
    gaussian_blur = cv.GaussianBlur(src = imgs, ksize= (3, 3), sigmaX = 0)
    # define low & hight threshold values for canny()
    low_threshold, high_threshold = 10, 30
    # apply canny edge detector function
    canny = cv.Canny(imgs, low_threshold, high_threshold)

    return canny

    # return cv.Canny(imgs, low_threshold, high_threshold)


# Define a function to processing images inside the directory
def image_processing(imgs):
    
    """ This functions'output will be the processed images and as input raw images
    from test_images/ directory """
    
    # gray scale
    gray_scale = cv.cvtColor(imgs, cv.COLOR_BGR2GRAY)
    # gaussian blur
    gaussian_blur = cv.GaussianBlur(src = imgs, ksize= (3, 3), sigmaX = 0)
    # define low & hight threshold values for canny()
    low_threshold, high_threshold = 10, 30
    # apply canny edge detector function
    canny = cv.Canny(imgs, low_threshold, high_threshold)

    return canny



def region_of_interest(imgs, vertices):
    # define a blank mask
    mask = np.zeros_like(imgs)   
    if len(imgs.shape) > 2:
        # i.e. 3 or 4 depending on your image
        channel_count = imgs.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon defined by "vertices" with the fill color    
    fill_poly = cv.fillPoly(img = mask, pts = [vertices], color = ignore_mask_color)
    # Returning the image only where mask pixels are nonzero
    masked_image = cv.bitwise_and(src1 = imgs, src2 = mask)
    # return masked_image and mask when calling the function
    return masked_image, mask

def hough_lines(canny, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Input `imgs` should be the output of a Canny transform. The function will then Return an image with hough lines drawn.
    """
    lines = cv.HoughLinesP(canny, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # lines = cv.HoughLines(imgs, rho, theta, threshold, min_line_len, max_line_gap)
    line_img = np.zeros((imgs.shape[0], imgs.shape[1], 3), dtype=np.uint8)
    # line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

'''
def hough_transform(imgs, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
'''

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
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
    # In case of error, don't draw the line
    draw_right, draw_left = True, True
    
    # Find slopes for every other line(s), however, select only those lines where abs(slope) > slope_threshold
    slope_threshold = 0.5
    slopes, new_lines = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
        
        # Calculate slope
        if x2 - x1 == 0.:  # corner case, avoiding division by 0
            slope = 999.  # practically infinite slope
        else:
            slope = (y2 - y1) / (x2 - x1)
            
        # Filter lines based on slope
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)
        
    lines = new_lines
    
    # Split right and left hand lane lines. Slope value will define whether we are dealing with a Left / Right Lane Line. 

    right_lines = []
    left_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        img_x_center = img.shape[1] / 2
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)
            
    # Using linear regression find the best line that fits for the right/ left lane.
    # Right lane lines
    right_lines_x = []
    right_lines_y = []
    
    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        
        right_lines_x.append(x1)
        right_lines_x.append(x2)
        
        right_lines_y.append(y1)
        right_lines_y.append(y2)
        
    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # from line equation y = m*x + b
    else:
        right_m, right_b = 1, 1
        draw_right = False
        
    # Left lane lines
    left_lines_x = []
    left_lines_y = []
    
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        
        left_lines_x.append(x1)
        left_lines_x.append(x2)
        
        left_lines_y.append(y1)
        left_lines_y.append(y2)
        
    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
    else:
        left_m, left_b = 1, 1
        draw_left = False
    
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)
    
    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m
    
    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m
    
    # Convert from float to int calculated points. 
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)
    
    # Draw right/ left lines on image
    if draw_right:
        cv.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if draw_left:
        cv.line(img, (left_x1, y1), (left_x2, y2), color, thickness)

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv.addWeighted(initial_img, α, img, β, γ)
        
        
def select_colors(image):
    """
    Filter the image to include only yellow and white pixels
    """
    # Filter white pixels
    white_threshold = 200
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv.inRange(image, lower_white, upper_white)
    white_image = cv.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    lower_yellow = np.array([90,100,100])
    upper_yellow = np.array([110,255,255])
    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv.bitwise_and(image, image, mask=yellow_mask)

    # Combine the two above images
    processed_image = cv.addWeighted(white_image, 1., yellow_image, 1., 0.)

    return processed_image


def annotate_image(imgs):
    """ Given an image Numpy array, return the annotated image as a Numpy array """
    # Only keep white and yellow pixels in the image, all other pixels become black
    image = select_colors(imgs)
    
    # Read in and grayscale the image
    #image = (image*255).astype('uint8')  # this step is unnecessary now
    gray = grayscale(image)

    # Apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, kernel_size)

    # Apply Canny Edge Detector
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Create masked edges using trapezoid-shaped region-of-interest
    imshape = image.shape
    vertices = np.array([[\
        ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),\
        ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
        (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
        (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]]\
        , dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Run Hough on edge detected image
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    # Draw lane lines on the original image
    initial_image = imgs.astype('uint8')
    annotated_image = weighted_img(line_image, initial_image)
    
    return annotated_image


# Display an example image
# annotated_image = annotate_image(imgs)

# Display image result
# plt.imshow('image', annotated_image)
cv.imshow(annotate_image(imgs))

cv.waitKey(0)
cv.destroyAllWindows()

