# Make a function to split a segment classification by label
# Split AB by label in column O
# Reference the process_race spreadsheet. Each video has a unique name. There is one unique AB segment for each video. Put the name_AB videos with positive labels in positiveO, and those with negative in negativeO
#%%
import os
import ast
import datetime
import argparse
import pandas as pd
from shutil import copyfile

# Custom imports
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings3


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

    destination_positive = os.path.join(args.output_directory, "positive"+args.segments)
    destination_negative = os.path.join(args.output_directory, "negative"+args.segments)
    if not os.path.exists(destination_positive):
        os.makedirs(destination_positive)

    if not os.path.exists(destination_negative):
        os.makedirs(destination_negative)

    for nn in range(len(df)):
        if len(args.segments == 2):
            if df.loc[nn]["timepoint_"+args.segments[0]] and df.loc[nn]["timepoint_"+args.segments[1]]:
                fname = 'flownet_%s_%s_%02d.mp4' % (df.iloc[nn]['meta_video_file_name'][:-4], args.segments, df.iloc[nn]['meta_position_nn'])
                video_input_fn = os.path.join(args.data_directory, args.segments, "optical_flow", fname)
                if not os.path.exists(video_input_fn):
                    continue
                label = df.iloc[nn][args.label]
                if label == 0:
                    video_output_fn = os.path.join(destination_negative, fname)
                else:
                    video_output_fn = os.path.join(destination_positive, fname)
                copyfile(video_input_fn, video_output_fn)
        elif len(args.segments == 1):
            if df.loc[nn]["timepoint_"+args.segments[0]]:
                timepoint = df.loc[nn]["timepoint_"+args.segments[0]]
                fname = 'flownet_%s_%s_%02d.mp4' % (df.iloc[nn]['meta_video_file_name'][:-4], args.segments, df.iloc[nn]['meta_position_nn'])
                video_input_fn = os.path.join(args.data_directory, args.segments, "optical_flow", fname)
                if not os.path.exists(video_input_fn):
                    continue
                # make this adapt to use max padding, finding the min between the set value and hte distance to the neighboring timepoint
                start_val = timepoint - 1
                end_val = timepoint + 1
                start_time = '0' + str(datetime.timedelta(seconds=start_val))
                n_frames = round(30.*(end_val-start_val))
                label = df.iloc[nn][args.label]
                if label == 0:
                    video_output_fn = os.path.join(destination_negative, fname)
                else:
                    video_output_fn = os.path.join(destination_positive, fname)
                cmd = 'ffmpeg -ss %s -i %s -an -vcodec h264 -r 30 -vframes %d %s' % (start_time, video_input_fn, n_frames, video_output_fn)
                os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input argument
    # parser.add_argument("--data_labels", default = settings3.data_labels, help = "Path to labels")
    parser.add_argument("--data_labels", default = "RACE_python_format_final.xlsx", help = "Path to labels")
    parser.add_argument("--data_directory", default = settings3.data_directory, help = "Path to the data directory")
    parser.add_argument("--output_directory", default = settings3.output_directory, help = "Path to the output directory")
    # parser.add_argument("--segments", default = "AB", help = "segments")
    arser.add_argument("--segments", default = "B", help = "segments")
    parser.add_argument("--label", default = "label_needle positionB", help = "label")
    parser.add_argument('--inclusive', dest='inclusive', action='store_true', help = "if inclusive, use first index of start, last index of end for list timepoints")
    args = parser.parse_args()
    splice(args)

