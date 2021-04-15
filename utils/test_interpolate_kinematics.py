import os
import glob
import cv2
import ast
from tqdm import tqdm
import datetime
import numpy as np
import pandas as pd
# Custom imports
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings


def interpolate():
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

def splice(all_timepoints = ['A', 'B', 'C', 'D', 'E', 'F', 'G']):
    df = get_labels(settings.data_labels)

    # Convert the strings to lists and none literals
    for ii in range(len(df)):
        for col_name in df.columns:
            if 'label' in col_name or 'timepoint' in col_name:
                curr_entry = df.at[ii, col_name]
                if type(curr_entry) == str:
                    #print(curr_entry, ii)
                    if curr_entry == 'none':
                        df.at[ii, col_name] = ast.literal_eval('None')
                    else:
                        df.at[ii, col_name] = ast.literal_eval(curr_entry)

    for nn in tqdm(range(len(df))):
        kk = 0
        kinematics_file = df.iloc[nn]["meta_raw_kinematic_data_name"]
        if kinematics_file is None or not os.path.exists(kinematics_file):
            print(kinematics_file)
            continue
        kinematics = pd.read_csv(kinematics_file)
        header = ','.join(list(kinematics.columns))
        kinematics = kinematics.values
        while kk < len(all_timepoints)-1:
            start_col = 'timepoint_%s' % all_timepoints[kk]
            end_col = 'timepoint_%s' % all_timepoints[kk+1]
            start_index = 0
            end_index = -1
            start_val, end_val = df.iloc[nn][start_col], df.iloc[nn][end_col]
            if start_val is None or end_val is None:
                #print('ff 2 due to None val')
                kk+=2
                pass
            else:
                # If start_val is a list make it the last one
                if type(start_val) == list:
                    start_val = start_val[start_index]
                if type(end_val) == list:
                    end_val = end_val[end_index]
                start_val = round(29.97*start_val)
                end_val = round(29.97*end_val)
                kinematics_splice = kinematics[start_val: end_val+1]
                output_subdir = os.path.join(settings.kinematics_directory, f"{all_timepoints[kk]}{all_timepoints[kk+1]}")
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                np.savetxt(os.path.join(output_subdir, os.path.basename(kinematics_file)), kinematics_splice, delimiter=",", header=header, comments='')
                kk+=1

if __name__ == '__main__':
    #interpolate()
    #splice()
    splice(all_timepoints=['A', 'C', 'E'])
