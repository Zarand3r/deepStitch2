import cv2
import numpy as np
import os
import glob
import argparse
# Custom imports
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings1


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
	parser = argparse.ArgumentParser(description='Training')
	parser.add_argument('--datadir', default=settings1.data_directory, help='input directory')
	parser.add_argument('--include_classes', default='AC_CE_EF_FG', type=str, help='Which classnames to include seperated by _ e.g. 00_01')
	hparams = parser.parse_args()

	datadir = hparams.datadir
	classes = hparams.include_classes.split('_')

	for c in classes:
		input_directory = os.path.join(datadir, c, "optical_flow")
		output_directory = os.path.join(datadir, c, "features")
		convert_all(input_directory, output_directory)


#	convert_all("../../../../balint_data_sample/CE/optical_flow", "../../../../balint_data_sample/test")
