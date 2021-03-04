import cv2
import numpy as np
import os
import glob
import argparse
import git
import sys
import pandas as pd
from ast import literal_eval
from collections import defaultdict

# Custom imports
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings

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

        while (fc < frameCount  and ret):
                try:
                        ret, buf[fc] = cap.read()
                except:
                        print("stopped at ", fc)
                fc += 1

        cap.release()
        np.save(output_path, buf)

def get_labels(labelsfile, header=[0]):
        raw_data = []
        try:
                raw_data = pd.read_excel(labelsfile, skiprows=0, header=header)
                print("Loaded excel file")
        except:
                try:
                        raw_data = pd.read_csv(labelsfile, skiprows=0, header=header)
                        print("Loaded csv file")
                except:
                        print("File must be in excel or csv format")
        raw_data = raw_data.where(pd.notnull(raw_data), None)
        return raw_data

def generate_labels(video_directory, output_directory, labels, stride=1):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # some videos dont have A, and start at a later marker. Keep this in mind.
        columns = ['timepoint_A', 'timepoint_B', 'timepoint_C', 'timepoint_D', 'timepoint_E', 'timepoint_F', 'timepoint_G']
        fps = 29.97/stride
        previousname = None
        crop_intervals = {}
        previous_end = 0
        for index, row in labels.iterrows():
                name = row["meta_video_file_name"]
                fname = os.path.join(video_directory, name[:-4]+".npy")
                if not os.path.exists(fname):
                        continue
                print(fname)
                frames = []
                intervals = []

                for col in columns:
                        t = row[col]
                        if t is None:
                                continue
                        if isinstance(t, str):
                                t = literal_eval(t)
                        point = col.split('_')[-1]
                        if isinstance(t, list):
                                for i in range(len(t)):
                                        frames.append(int(t[i]*fps))
                                        intervals.append(point)
                        else:
                                frames.append(int(t*fps))
                                intervals.append(point)

                if len(frames) == 1:
                        continue

                result = {}
                result = defaultdict(lambda:0,result)
                classes_list = []
                for i in range(len(frames)-1):
                        classification = intervals[i] + intervals[i+1]
                        result[classification] += frames[i+1] - frames[i]
                        classes_list.append(classification)


                if name == previousname:
                        with open(os.path.join(output_directory, f'{name[:-4]}.txt'), 'a+') as output_file:
                                if frames[0] > previous_end:
                                        for i in range(frames[0]-previous_end):
                                                output_file.write('Other\n')
                                # for classification in classes_list:
                                for classification in result:
                                        for i in range(result[classification]):
                                                output_file.write('%s\n' % classification)

                else:
                        with open(os.path.join(output_directory, f'{name[:-4]}.txt'), 'w') as output_file:
                                # for classification in classes_list:
                                for classification in result:
                                        for i in range(result[classification]):
                                                output_file.write('%s\n' % classification)

                previousname = name
                previous_end = frames[-1]
                if fname in crop_intervals:
                    crop_intervals[fname][1] = previous_end
                else:
                    crop_intervals[fname] = [frames[0], previous_end]

        for fname in crop_intervals:
                video = np.load(fname)
                start = crop_intervals[fname][0]
                end = crop_intervals[fname][1]
                if len(video) != (end-start):
                    print(f"{len(video)} changed to {end-start}")
                    video = video[start:end]
                    np.save(fname, video.T)
        return
# set upper range to result[classification]-5*2

if __name__ == '__main__':
        # convert_video("2020_05_08_14_06_20_CUT.mp4","")
        labels = get_labels(settings.data_labels)
        generate_labels("/mnt/md1/richard_bao/balint_data/TCN_data/features","/mnt/md1/richard_bao/balint_data/TCN_data/groundTruth",labels, 2)
