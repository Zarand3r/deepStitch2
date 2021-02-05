import cv2
import numpy as np
import time
import argparse
import os

# Instead of replacing with background, replace with monoocolor
# Detect circles within range of radius for validation

def cloak(args):
	# background_cap = 
	cap = cv2.VideoCapture(args.video if args.video else 0)

	# We give some time for the camera to setup
	time.sleep(3)
	count = 0
	background=0

	# Capturing and storing the static background frame
	video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	num_iterations = min(int(video_length)/3, 30)
	for i in range(video_length):
		ret,background = cap.read()
	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

	#background = np.flip(background,axis=1)

	while(cap.isOpened()):
		ret, img = cap.read()
		if not ret:
			break
		count+=1
		#img = np.flip(img,axis=1)
		
		# Converting the color space from BGR to HSV
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		# print(hsv[570][575])

		# # Generating mask to detect red color
		# lower_green = np.array([25, 52, 72])
		# upper_green = np.array([102, 255, 255])
		# mask1 = cv2.inRange(hsv,lower_green,upper_green)

		# lower_yellow = np.array([60, 85, 90]) #[61, 86.4, 92.5]
		# upper_yellow = np.array([62, 87, 95])
		# mask2 = cv2.inRange(hsv,lower_yellow,upper_yellow)

		# mask1 = mask1+mask2

		lower_yellow = np.array([28, 200, 200]) #[61, 86.4, 92.5]
		upper_yellow = np.array([33, 230, 240])
		mask1 = cv2.inRange(hsv,lower_yellow,upper_yellow)

		# Refining the mask corresponding to the detected red color
		mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
		mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations = 1)
		mask2 = cv2.bitwise_not(mask1)

		# Generating the final output
		res1 = cv2.bitwise_and(background,background,mask=mask1)
		res2 = cv2.bitwise_and(img,img,mask=mask2)
		final_output = cv2.addWeighted(res1,1,res2,1,0)

		cv2.imshow('final',final_output)
		k = cv2.waitKey(10)
		if k == 27:
			break

def cover_dot2(args):
	cap = cv2.VideoCapture(args.video if args.video else 0)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	frame_fps = int(cap.get(5))
	output_directory, output_fname = os.path.split(args.video)
	output_directory += "_masked"
	output_path = os.path.join(output_directory, output_fname)
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
	result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), frame_fps, (frame_width,frame_height))
	time.sleep(3)

	if cap.isOpened():
		known_contour = []
		while(True):
			ret, frame = cap.read()
			if frame is not None and args.optical:
				optical = frame[:,int(frame_width/2):]
				frame = frame[:,:int(frame_width/2)]
			# blurring the frame that's captured
			frame_gau_blur = cv2.GaussianBlur(frame, (3, 3), 0)
			# converting BGR to HSV
			hsv = cv2.cvtColor(frame_gau_blur, cv2.COLOR_BGR2HSV)


			canny_edge = cv2.Canny(frame_gau_blur, 300, 500)
			contours, _ = cv2.findContours(canny_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			contour_list = []
			for contour in contours:
				approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
				area = cv2.contourArea(contour)
				if (area > 20):
					contour_list.append(contour)
			# TODO: keep track and store the largest contour that makes the largest polygon in known_contour and append to contour_list every iteration
			cv2.fillPoly(frame,pts=contour_list,color=(0,0,0))
			# contours = sorted(contours, key=lambda x: cv2.contourArea(x))
			# biggest= max(contours, key = cv2.contourArea)
			# cv2.drawContours(frame, biggest, -1, (0, 0, 255), thickness=-1)


			# update the with the results of the previous masking
			frame_gau_blur = cv2.GaussianBlur(frame, (3, 3), 0)
			hsv = cv2.cvtColor(frame_gau_blur, cv2.COLOR_BGR2HSV)
			lower_green = np.array([50, 210, 210])
			higher_green = np.array([70, 250, 270])
			lower_yellow = np.array([28, 200, 200]) #[61, 86.4, 92.5]
			higher_yellow = np.array([33, 230, 240])
			# getting the range of yellow color in frame
			green_range = cv2.inRange(hsv, lower_green, higher_green)
			yellow_range = cv2.inRange(hsv, lower_yellow, higher_yellow)
			kernel = np.ones((10,10),np.uint8)
			green_range = cv2.dilate(green_range,kernel,iterations = 3)
			yellow_range = cv2.dilate(yellow_range,kernel,iterations = 3)
			combined_range = green_range + yellow_range
			# res_combined = cv2.bitwise_and(frame_gau_blur,frame_gau_blur, mask=combined_range)
			frame[combined_range>0] = (0,0,0)

			# applying HoughCircles
			# circles = cv2.HoughCircles(canny_edge, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=10, param2=20, minRadius=1, maxRadius=50)
			# print(circles)
			# cir_cen = []
			# if circles is not None:
			# 	# circles = np.uint16(np.around(circles))
			# 	for i in circles[0,:]:
			# 		# drawing on detected circle and its center
			# 		cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
			# 		cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
			# 		cir_cen.append((i[0],i[1]))
			# print(cir_cen)
			# contours, _ = cv2.findContours(canny_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			# cv2.fillPoly(frame,pts=contours,color=(0,0,0))
			if args.optical:
				frame = np.concatenate((frame,optical), axis=1)		
			result.write(frame)	
			if args.visualize:
				cv2.imshow('final',frame)
			k = cv2.waitKey(10)
			if k == 27:
				break
		cv2.destroyAllWindows()
	else:
		print('no cam')
		return

def cover_dot(args):
	cap = cv2.VideoCapture(args.video if args.video else 0)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	frame_fps = int(cap.get(5))
	output_directory, output_fname = os.path.split(args.video)
	output_directory += "_masked"
	output_path = os.path.join(output_directory, output_fname)
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
	result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), frame_fps, (frame_width,frame_height)) 

	# We give some time for the camera to setup
	time.sleep(3)
	count = 0

	#background = np.flip(background,axis=1)

	while(cap.isOpened()):
		ret, img = cap.read()
		if not ret:
			break
		count+=1
		#img = np.flip(img,axis=1)

		if args.optical:
			optical = img[:,int(frame_width/2):]
			img = img[:,:int(frame_width/2)]
		
		# Converting the color space from BGR to HSV
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		# Generating mask to detect red color
		lower_green = np.array([50, 220, 230])
		upper_green = np.array([60, 240, 250])
		mask1 = cv2.inRange(hsv,lower_green,upper_green)

		lower_yellow = np.array([28, 190, 200]) #[61, 86.4, 92.5]
		upper_yellow = np.array([33, 230, 240])
		mask2 = cv2.inRange(hsv,lower_yellow,upper_yellow)

		# Refining the mask corresponding to the detected red color
		mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
		mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations = 10)
		mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
		mask2 = cv2.dilate(mask2,np.ones((3,3),np.uint8),iterations = 10)
		img[mask1 == 255] = [0, 0, 0]
		img[mask2 == 255] = [0, 0, 0]

		output_img = img
		if args.optical:
			output_img = np.concatenate((img,optical), axis=1)		
		result.write(output_img)	
		if args.visualize:
			cv2.imshow('final',img)
		k = cv2.waitKey(10)
		if k == 27:
			break

	cap.release()
	result.release()
	cv2.destroyAllWindows() 


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Input argument
	parser.add_argument("--video", default = "test_videos/test3.m4v", help = "Path to input video file. Skip this argument to capture frames from a camera.")
	parser.add_argument("--mode", default = "2", help = "Enter 1 to mask with background. Enter 2 to mask with a black cover")
	parser.add_argument("--optical", dest='optical', action='store_true', help = "Is the input video a split screen with optical flow?")
	parser.add_argument("--visualize", dest='visualize', action='store_true', help = "see the masked video frame by frame")
	args = parser.parse_args()
	if args.mode == 1:
		cloak(args)
	else:
		# cover_dot(args)
		cover_dot2(args)
