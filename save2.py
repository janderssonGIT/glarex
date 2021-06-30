# USAGE
# python light_detection_stream.py

# EDIT CAFFEMODEL och RENSA de funktioner du ej behöver. TESTA

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

def interpolate(inputframe):

  # Get input size
	height, width = inputframe.shape[:2]

  # Desired "pixelated" size
	w, h = (70, 120) # 70x120 is the cell size of the prototype sheet

  # Resize input to "pixelated" size	
	temp = cv2.resize(inputframe, (w, h), interpolation=cv2.INTER_LINEAR)

  # Initialize output image
	output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

	return output

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()

  #Make grayscale
	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grayframe)
	cv2.circle(grayframe, maxLoc, 5, (255, 0, 0), 2)

	# Apply a Gaussian blur to the image then find the brightest region
	grayframe = cv2.GaussianBlur(grayframe, (41, 41), 0)
 
	#TODO Make a center rectangle
	cv2.circle(frame, maxLoc, 41, (255, 0, 0), 2)

	# Pixelate the frame
	grayframe = interpolate(grayframe)

	# Show the output frame
	cv2.imshow("Frame", grayframe)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
