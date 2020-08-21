# python detect_barcode_opencv.py --image images/barcode_01.jpg

# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image file")
ap.add_argument("--show", help = "option to show inner images", type=int)

args = vars(ap.parse_args())
show = args["show"]

# load the image and convert it to grayscale
image = cv2.imread(args["image"])

#resize image
image = cv2.resize(image,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)

#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#calculate x & y gradient
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
if show == 1:
	cv2.imshow("gradient-sub",cv2.resize(gradient,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))

# blur the image
blurred = cv2.blur(gradient, (3, 3))

# threshold the image
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

if show == 1:
	cv2.imshow("threshed",cv2.resize(thresh,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

if show == 1:
	cv2.imshow("morphology",cv2.resize(closed,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

if show == 1:
	cv2.imshow("erode/dilate",cv2.resize(closed,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts,hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
c1 = sorted(cnts, key = cv2.contourArea, reverse = True)[1]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
rect1 = cv2.minAreaRect(c1)
box1 = np.int0(cv2.boxPoints(rect1))

# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.drawContours(image, [box1], -1, (0, 255, 0), 3)

image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

cv2.imshow("Image", image)
cv2.waitKey(0)
