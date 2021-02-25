import numpy as np
import pandas as pd
import os

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

def get_kinematics(kinematicsfile):
	with open(kinematicsfile) as f:
		for line in f:
			# make csv between time points


def main(labelsfile, kinematics_path):
	labels = get_labels(labelsfile)
	for index, row in labels.iterrows():
		if index > 0:
			break
		kinematicsfile = row["meta_raw_kinematic_data_name"]
		kinematicsfile = os.path.join(kinematics_path, kinematicsfile)
		kinematics = get_kinematics(kinematicsfile)

if __name__ == '__main__':
	kinematics_directory = ""
	labelsfile = "../../../RACE_python_format_final.csv"
	main(labelsfile)

