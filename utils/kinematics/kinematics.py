# Given python labels file and kinematics directory, process kinematics data into a numpy file named after the corresponding video, and output in a kinematics folder on the same level as original
# in the lightning_train CustomDataset __getitem__ method return the kinematics too. Modify the training_step because batch is now a 3-tuple.
# input_cuda, target_cuda = self.apply_transforms_GPU(batch[0:2], random_crop=self.hparams.random_crop)
# the kinematics is just a label. The forward function still just takes input_cuda.
# Do the old stuff with first two elements (input_cuda, target_cuda), and use the third (kinematics) for target_kinematics

import numpy as np
import pandas as pd

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

if __name__ == '__main__':
	labels_file = "../../../RACE_python_format_final.csv"
	foo = get_labels(labels_file)
	print(foo.head)

