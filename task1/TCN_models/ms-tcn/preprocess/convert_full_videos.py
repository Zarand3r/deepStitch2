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


def convert_video(video_path, output_directory):
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
        while (fc < frameCount and ret):
            try:
                ret, buf[fc] = cap.read()
            except:
                print("stopped at: ", fc)
            fc += 1
        cap.release()
        np.save(output_path, buf)

def convert_all(input_directory, output_directory=None):
        if not os.path.exists(output_directory):
                os.makedirs(output_directory)
        #for file in glob.glob(os.path.join(input_directory, "*.mp4")):
        #    convert_video(file, output_directory)
        for file in glob.glob(os.path.join(input_directory, "*.m4v")):
            convert_video(file, output_directory)

#def label_video()

#def label_all(input_directory, output_directory=None):

if __name__ == '__main__':
        # do stuff
        parser = argparse.ArgumentParser(description='Training')
        parser.add_argument('--rawdata', default=settings1.raw_directory, help='directory to get raw data')
        parser.add_argument('--datadir', default=settings1.data_directory, help='directory to write numpy data features')
        hparams = parser.parse_args()

        rawdata = hparams.rawdata
        datadir = hparams.datadir

        input_directory = rawdata
        output_directory = os.path.join(datadir, "TCN", "features")
        convert_all(input_directory, output_directory)


#       convert_all("../../../../balint_data_sample/CE/optical_flow", "../../../../balint_data_sample/test")
