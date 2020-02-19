# **Finding Lane Lines on the Road** 

## Writeup Template


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

In working for this project to testing the solution pipeline, the images used were the following:

[CarND-LaneLines-P1/test_images/]: # (Image References)
['solidWhiteRight.jpg',
 'solidWhiteCurve.jpg',
 'solidYellowCurve2.jpg',
 'solidYellowLeft.jpg',
 'whiteCarLaneSwitch.jpg',
 'solidYellowCurve.jpg']
 
[image1]: ./examples/grayscale.jpg "Grayscale"




### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the RGB images to grayscale. To reducing the noise right upfront of the pipeline, Gaussian blur was performed followed by canny edge detector to detect the edges in the image. Next, a triangular region of interest was defined in order to confine the detection algorithm thus, avoiding detecting other objects/noises. Having, entailed the area of interest through coding, next lines were detected followed by Hough transform to make the correct approximation of the correct and most line(s). Finally draw lines and weighted line were used in order to show the results in the image and/videos.  


### 2. Identify potential shortcomings with your current pipeline

Potential shortcomings

-Strongly curved lane lines
-Variations of brighness and light conditions can affect detection of lines.
-Shadows casted by the presence of other objects nearby the road 



### 3. Suggest possible improvements to your pipeline

Tweaking further Canny edge detector could better improve the accuracy of detection the edges resulting in better line detection. Futhermore, reducing image noises through filtering could be another solution to improve the detection pipeline.

