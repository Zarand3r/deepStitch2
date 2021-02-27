# Given python labels file and kinematics directory, process kinematics data into a csv file named after the corresponding video, and output in a kinematics folder on the same level as original
# in the lightning_train CustomDataset __getitem__ method return the kinematics too. Modify the training_step because batch is now a 3-tuple.
# input_cuda, target_cuda = self.apply_transforms_GPU(batch[0:2], random_crop=self.hparams.random_crop)
# the kinematics is just a label. The forward function still just takes input_cuda.
# Do the old stuff with first two elements (input_cuda, target_cuda), and use the third (kinematics) for target_kinematics
import os
import csv
import datetime
import numpy as np
import pandas as pd
from ast import literal_eval

# Custom imports
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings1


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

def get_kinematics(kinematics_directory, kinematicsfile, filename, start, end):
        rows = []
        dict = {}
        try:
            with open(os.path.join(kinematics_directory, kinematicsfile)) as f:
                    for line in f:
                            label = line.split(":")[0].replace(" ", "_")
                            if label == "Time_Stamp":
                                    if len(dict) > 0:
                                            rows.append(dict)
                                    dict = {}
                                    timepoint = float(line.split(":")[-1].strip())
                                    timepoint = timepoint - start
                                    dict[label] = timepoint
                                    continue
                            if timepoint < 0:
                                    dict = {}
                                    continue
                            elif timepoint > end-start:
                                    break
                            else:
                                    dict[label] = float(line.split(":")[-1].strip())
        except:
            print(f"Log file {kinematicsfile} does not exist")

        output_subdir = os.path.join(kinematics_directory, "demonstrations")
        if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

        kinematics_output_fn = os.path.join(kinematics_directory, "demonstrations", filename+".csv")
        if len(rows) > 0:
            with open(kinematics_output_fn, 'w') as csvfile: 
                    writer = csv.DictWriter(csvfile, fieldnames = (rows[0].keys()))  
                    writer.writeheader()  
                    writer.writerows(rows) 
        return kinematics_output_fn

def main(labelsfile, data_directory, kinematics_directory):
        labels = get_labels(labelsfile)
        previous_filename = None
        count = 1
        for index, row in labels.iterrows():
                filename = row['meta_video_file_name']
                if filename == previous_filename:
                        count += 1
                else:
                        count = 1
                previous_filename = filename
                kinematicsfile = row["meta_raw_kinematic_data_name"]
                timepoints = list(row[["timepoint_A", "timepoint_B", "timepoint_C", "timepoint_D", "timepoint_E", "timepoint_F", "timepoint_G"]])
                timepoints = [t for t in timepoints if t is not None]
                start = literal_eval(str(timepoints[0]))
                end = literal_eval(str(timepoints[-1]))
                if isinstance(start, list):
                        start = start[0]
                if isinstance(end, list):
                    end = end[-1]

                video_input_fn = os.path.join(data_directory, filename)
                output_subdir = os.path.join(data_directory, "demonstrations")
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                video_output_fn = os.path.join(output_subdir, f"{filename[:-4]}_{count}.mp4") #m4v becomese mp4
                start_time = '0' + str(datetime.timedelta(seconds=start))
                n_frames = round(30.*(end-start))
                cmd = 'ffmpeg -ss %s -i %s -an -vcodec h264 -r 30 -vframes %d %s' % (start_time, video_input_fn, n_frames, video_output_fn)
                os.system(cmd)
                kinematics_output_fn = get_kinematics(kinematics_directory, kinematicsfile, f"{filename[:-4]}_{count}", start, end)
                labels.at[index, 'meta_video_file_name'] = video_output_fn
                labels.at[index, 'meta_raw_kinematic_data_name'] = kinematics_output_fn
                for time_label in ["timepoint_A", "timepoint_B", "timepoint_C", "timepoint_D", "timepoint_E", "timepoint_F", "timepoint_G"]:
                        t = labels.at[index, time_label]
                        if t is None:
                                continue
                        t = literal_eval(str(t))
                        if isinstance(t, list):
                                for i in range(len(t)):
                                        t[i] -= start
                        else:
                                t -= start
                        labels.at[index, time_label] = t

        labels.to_csv(f"{labelsfile[:-4]}_demonstrations.csv", index=False)

if __name__ == '__main__':
        labelsfile = settings1.data_labels
        kinematics_directory = settings1.kinematics_directory
        data_directory = settings1.raw_directory
        main(labelsfile, data_directory, kinematics_directory)


