#%%
import pandas as pd
import os
import argparse

# Custom imports
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings1


# Load the preprocessed input video
# If it has more frames than the sliding window, iteratively slice and classify each slice
# Return a list of classifications per frame. The starting point of the slice that has a different label than before is the beginning of the new label.


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input argument
    parser.add_argument("--data_labels", default = settings1.data_labels, help = "Path to labels")
    parser.add_argument("--raw_directory", default = settings1.raw_directory, help = "Path to the raw data ")
    parser.add_argument("--data_directory", default = settings1.data_directory, help = "Path to the output directory")
    args = parser.parse_args()
    splice(args)
