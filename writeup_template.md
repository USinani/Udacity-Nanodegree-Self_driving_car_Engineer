# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the BGR colored images to grayscale, then I
moved on to converting the color space to HSV.



In order to draw a single line on the left and right lanes, I modified the draw_lines() function by using the linear approximation equation.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ...

-light intensity changes
-area of interest is not clear enough to detect the regions with hight color contrast
-Noises present in the image

Another shortcoming could be.
-camera lens distortion


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use better image filters

Another potential improvement could be to better correct for distortion
