import cv2
import numpy as np
import os
import glob

def convert_file(video_path, output_directory):
	print(video_path)
	cap = cv2.VideoCapture(video_path)
	vname = os.path.split(video_path)[-1][:-4]
	output_path = os.path.join(output_directory, vname)
	frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

	fc = 0
	ret = True

	while (fc < frameCount  and ret):
		ret, buf[fc] = cap.read()
		fc += 1

	cap.release()
	np.save(output_path, buf)

def convert_all(input_directory, output_directory=None):
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
	for file in glob.glob(os.path.join(input_directory, "*.mp4")):
	    convert_file(file, output_directory)

if __name__ == '__main__':
	# do stuff
	convert_all("../../../../balint_data_sample/CE/optical_flow", "../../../../balint_data_sample/test")