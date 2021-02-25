# Given python labels file and kinematics directory, process kinematics data into a csv file named after the corresponding video, and output in a kinematics folder on the same level as original
# in the lightning_train CustomDataset __getitem__ method return the kinematics too. Modify the training_step because batch is now a 3-tuple.
# input_cuda, target_cuda = self.apply_transforms_GPU(batch[0:2], random_crop=self.hparams.random_crop)
# the kinematics is just a label. The forward function still just takes input_cuda.
# Do the old stuff with first two elements (input_cuda, target_cuda), and use the third (kinematics) for target_kinematics

import numpy as np
import pandas as pd
import os
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

def get_kinematics(kinematicsfile, start, end):
	with open(kinematicsfile) as f:
		for line in f:
			if "Time Stamp" in line:
				timepoint = float(line.split(":")[-1].strip())
			if timepoint < start:
				continue
			elif timepoint > end:
				break
			else:
				label = line.split(":")[0].replace(" ", "_")
				print(label)


def main(labelsfile, kinematics_directory):
	labels = get_labels(labelsfile)
	for index, row in labels.iterrows():
		if index > 0:
			break
		kinematicsfile = row["meta_raw_kinematic_data_name"]
		timepoints = list(row[["timepoint_A", "timepoint_B", "timepoint_C", "timepoint_D", "timepoint_E", "timepoint_F", "timepoint_G"]])
		timepoints = [t for t in timepoints if t is not None]
		start = literal_eval(str(timepoints[0]))
		end = literal_eval(str(timepoints[-1]))
		if isinstance(start, list):
			start = start[0]
		kinematicsfile = os.path.join(kinematics_directory, kinematicsfile)
		kinematics = get_kinematics(kinematicsfile, start, end)

if __name__ == '__main__':
	kinematics_directory = "/Users/richardbao/Research/anima/robosurgery/kinematic_data"
	labelsfile = "../../../RACE_python_format_final.csv"
	main(labelsfile, kinematics_directory)

