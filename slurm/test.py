import os
import cv2


for path, subdirs, files in os.walk("/central/groups/tensorlab/rbao/robosurgery/balint_data/classification_data/positive"):
	for name in files:
		fname = os.path.join(path, name)
		cap = cv2.VideoCapture(fname)
		length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if length < 10:
			os.remove(fname)


for path, subdirs, files in os.walk("/central/groups/tensorlab/rbao/robosurgery/balint_data/classification_data/negative"):
	for name in files:
		fname = os.path.join(path, name)
		cap = cv2.VideoCapture(fname)
		length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if length < 10:
			os.remove(fname)

