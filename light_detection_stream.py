# USAGE
# in (cv) python light_detection_stream.py

# Issues:
# TODO If no light source, keep running feed! Fix this loop
# TODO Make resolution always native to camera

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

MIN_INTENSITY = 205
MAX_INTENSITY = 255

# Begin video stream, allow the camera sensor to warm up
# Start the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
	
def interpolateDown(inputframe):
	return cv2.resize(inputframe, (120, 70), interpolation=cv2.INTER_AREA)

def interpolateUp(inputFrame):
	return cv2.resize(pixellatedframe, (1280, 720), interpolation=cv2.INTER_NEAREST) 

def thresholdmask(inputframe):
	thresh = cv2.threshold(inputframe, MIN_INTENSITY, MAX_INTENSITY, cv2.THRESH_BINARY)[1] 				# Threshold the image to reveal lighter regions
	thresh = cv2.erode(thresh, None, iterations=0) 	#2																						# Perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
	thresh = cv2.dilate(thresh, None, iterations=0)	#4
 
	cv2.imshow("Mask", thresh)																																		# Show video output of masking (DEBUG PURPOSE)

	labels = measure.label(thresh, background=0) 																									# Perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
	mask = np.zeros(thresh.shape, dtype="uint8")

	for label in np.unique(labels): 																															# Loop over the unique components
		if label == 0: 																																							# If this is the background label, ignore it
			continue
		labelMask = np.zeros(thresh.shape, dtype="uint8") 																					# Otherwise, construct the label mask and count the number of pixels 
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)

		if numPixels > 3 and numPixels < 10000: 																											# If the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
			mask = cv2.add(mask, labelMask)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)							# Find the contours in the mask, then sort them from left to right
	cnts = imutils.grab_contours(cnts)
	cnts = contours.sort_contours(cnts)[0]
 
	return cnts

# Loop over the frames from the video stream
# MAIN LOOP
while True:
	frame = vs.read() 																															# Read a frame from videostream
	height, width = frame.shape[:2] 																								# Camera resolution

	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 														# Make the frame grayscale
	blurredframe = cv2.GaussianBlur(grayframe, (41, 41), 0) 												# Apply blur to the greyed frame
	pixellatedframe = interpolateDown(blurredframe)
	cnts = thresholdmask(pixellatedframe) 																					# Detect the brightest regions and return the contours of the frame

	#pixellatedframe = cv2.merge([pixellatedframe,pixellatedframe,pixellatedframe])
	#pixellatedframe = cv2.cvtColor(pixellatedframe,cv2.COLOR_GRAY2RGB)
	for (i, c) in enumerate(cnts):																									# Now we draw a circle around the bright areas in the contour image
		cv2.fillPoly(pixellatedframe, pts = [c], color=(0,128,0))
		#c2 = np.array(c).flatten()
		print("x, y coords for blob #", i)
		print(*c, sep = ", ") 
		print("\n")

		(x, y, w, h) = cv2.boundingRect(c)																						# Draw the bright spot on the image
		cv2.putText(pixellatedframe, "#{}".format(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.15, (0, 0, 255), 1) #25 is number of detections

	pixellatedframe = interpolateUp(pixellatedframe)
	cv2.imshow("Output", pixellatedframe)																						# Show the output frame
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
