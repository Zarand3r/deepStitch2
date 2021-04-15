import os
import glob
import cv2
import numpy as np
import pandas as pd
# Custom imports
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings

kinematics_file_list = glob.glob(os.path.join(settings.kinematics_directory, "*.csv"))
for kinematics_file in kinematics_file_list:
    video_path = os.path.join(settings.raw_directory, os.path.basename(kinematics_file[:-4])+".mp4")
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kinematics = pd.read_csv(kinematics_file)
    num_columns = len(kinematics.columns) #kinematics.shape[1]
    header = ','.join(list(kinematics.columns))
    kinematics = kinematics.values
    x = [n/29.97 for n in range(1, length+1)]
    xp = kinematics[:, 0]
    columns = [x]
    for i in range(1, num_columns):
        yp = kinematics[:, i]
        columns.append(np.interp(x, xp, yp))
    kinematics = np.column_stack(columns)
    np.savetxt(kinematics_file, kinematics, delimiter=",", header=header, comments='')
