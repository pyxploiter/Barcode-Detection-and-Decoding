import os
import argparse
import zbar
import numpy as np
import cv2

def preprocess(image):
	# load the image
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

	# blur the image
	blurred = cv2.blur(gradient, (3, 3))

	# threshold the image
	(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
	thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return thresh

	
def barcode(image):
	# create a reader
	scanner = zbar.ImageScanner()
	
	# configure the reader
	scanner.parse_config('enable')
	
	# obtain image data
	width, height = image.shape
	raw = image.tobytes()

	image = zbar.Image(width, height, 'Y800', raw)

	# scan the image for barcodes
	scanner.scan(image)
	
	# extract results
	for symbol in image:
	    # do something useful with results
	    print 'format:', symbol.type, '| data:', '"%s"' % symbol.data
	# clean up
	print '-----------------------------------------------------------------------'
	del(image)
	
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image file")
args = vars(ap.parse_args())
image = cv2.imread(args["image"],0) 
image = preprocess(args["image"])
barcode(image)
