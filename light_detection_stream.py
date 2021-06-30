# USAGE
# in (cv) python light_detection_stream.py

# TODO - PROBLEMS
# If no light source, keep running feed! Fix this loop

# Imports
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import time
import cv2

# Initialize video stream, allow the cammra sensor to warmup,
# and start the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

def interpolate(inputframe):

	height, width = inputframe.shape[:2] 																						# Get input size
	w, h = (70, 120) 																																# 70x120 is the cell size of the prototype sheet
	temp = cv2.resize(inputframe, (w, h), interpolation=cv2.INTER_LINEAR)   				# Resize input to "pixelated" size
	output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST) 		# Initialize output image

	return output

def thresholdmask(inputframe):
	print(inputframe.shape[:2])
	thresh = cv2.threshold(inputframe, 253, 255, cv2.THRESH_BINARY)[1] 						# Threshold the image to reveal lighter regions
	thresh = cv2.erode(thresh, None, iterations=2) 	#2															# Perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
	thresh = cv2.dilate(thresh, None, iterations=4)	#4

	labels = measure.label(thresh, background=0) 																		# Perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
	mask = np.zeros(thresh.shape, dtype="uint8")

	for label in np.unique(labels): 																								# Loop over the unique components
		
		if label == 0: 																																# If this is the background label, ignore it
			continue

		labelMask = np.zeros(thresh.shape, dtype="uint8") 														# Otherwise, construct the label mask and count the number of pixels 
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		#print(numPixels)

		if numPixels > 1500: 																													# If the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
			mask = cv2.add(mask, labelMask)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# Find the contours in the mask, then sort them from left to right
	cnts = imutils.grab_contours(cnts)
	cnts = contours.sort_contours(cnts)[0]
	#print("this is a countour", cnts)
	#exit()

	return cnts

# Loop over the frames from the video stream
# MAIN LOOP
while True:
	frame = vs.read() 																															# Read a frame from videostream

	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 														# Make the frame grayscale
	blurredframe = cv2.GaussianBlur(grayframe, (41, 41), 0) 												# Apply blur to the greyed frame
	pixellatedframe = interpolate(blurredframe)
	cnts = thresholdmask(pixellatedframe) 																					# Detect the brightest regions and return the contours of the frame
  
	for (i, c) in enumerate(cnts):																									# Now we draw a circle around the bright areas in the contour image

		(x, y, w, h) = cv2.boundingRect(c)																						# Draw the bright spot on the image
		((cX, cY), radius) = cv2.minEnclosingCircle(c)
		cv2.circle(pixellatedframe, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
		cv2.putText(pixellatedframe, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow("Frame", pixellatedframe)																						# Show the output frame
	key = cv2.waitKey(1) & 0xFF

	# If `q` key is pressed, break loop
	if key == ord("q"):
		break

	# Update FPS counter
	fps.update()

# Stop timer, display FPS
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
vs.stop()
