#%%
import pandas as pd
import ast
from tqdm import tqdm
import datetime
import os
import argparse

# Custom imports
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings1


def splice(args):   
    df = pd.read_excel(args.data_labels)
    df = df.where(pd.notnull(df), None)
    # make modular

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

    nn = 0
    all_timepoints = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    input_dir = args.raw_directory
    output_dir = args.data_directory

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xx in range(6):
        df['meta_cut_%s%s' % (all_timepoints[xx], all_timepoints[xx+1])] = None


    # First do all the cuts between different letters
    for nn in tqdm(range(len(df))):
        kk = 0
        while kk < len(all_timepoints)-1:
            start_col = 'timepoint_%s' % all_timepoints[kk]
            end_col = 'timepoint_%s' % all_timepoints[kk+1]
            #print('on %s to %s' % (start_col, end_col))
            start_val, end_val = df.iloc[nn][start_col], df.iloc[nn][end_col]
            if start_val is None or end_val is None:
                kk+=2
                pass
            else:
                # If start_val is a list make it the last one
                if type(start_val) == list:
                    start_val = start_val[-1]
                if type(end_val) == list:
                    end_val = end_val[0]
                #print(start_val)
                video_input_fn = df.iloc[nn]['meta_video_file_name']
                video_input_fn = os.path.join(input_dir, video_input_fn)
                video_output_fn = '%s_%s%s_%02d.mp4' % (df.iloc[nn]['meta_video_file_name'][:-4], all_timepoints[kk], all_timepoints[kk+1], df.iloc[nn]['meta_position_nn'])
                df.at[nn, 'meta_cut_%s%s' % (all_timepoints[kk], all_timepoints[kk+1])] = video_output_fn
                # video_output_fn = os.path.join(output_dir, video_output_fn)
                output_subdir = os.path.join(output_dir, f"{all_timepoints[kk]}{all_timepoints[kk+1]}")
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                video_output_fn = os.path.join(output_subdir, video_output_fn)
                if not os.path.exists(video_input_fn):
                    raise ValueError('The file you are trying to chop does not exist')
                else:
                    if not os.path.exists(video_output_fn):
                        print(video_input_fn)
                    kk+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input argument
    parser.add_argument("--data_labels", default = settings1.data_labels, help = "Path to labels")
    parser.add_argument("--raw_directory", default = settings1.raw_directory, help = "Path to the raw data ")
    parser.add_argument("--data_directory", default = settings1.data_directory, help = "Path to the output directory")
    args = parser.parse_args()
    splice(args)


