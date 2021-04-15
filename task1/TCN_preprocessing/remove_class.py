import glob
# Custom imports
import os
import git
import sys
import numpy as np
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings

# DONT FORGET TO REMOVE THE CLASS FROM THE mapping.txt file in the data directory too

def remove_class(label="Other"):
    # feature_files = glob.glob(f"{settings.TCN_data}/features/*.npy")
    feature_directory = os.path.join(settings.TCN_data, "features")
    label_files = glob.glob(f"{settings.TCN_data}/groundTruth/*.txt")
    label += "\n"
    for file in label_files:
        name = os.path.basename(file)
        features_file = os.path.join(feature_directory, name[:-4]+".npy")
        if not os.path.exists(features_file):
            continue
        discard_indices = []
        label_data = []
        with open(file) as labels:
            for i, l in enumerate(labels):
                if l == label:
                    discard_indices.append(i)
                else:
                    label_data.append(l)
        with open(file, 'w') as labels:
            for l in label_data:
                labels.write(l)

        features = np.load(features_file)
        features = features.T
        features = np.delete(features, discard_indices, axis=0)
        np.save(features_file, features.T)

if __name__ == '__main__':
        remove_class()
