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
import settings

#import convert_using_flownet

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

def splice(args):   
    df = get_labels(args.data_labels)

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

    if len(args.segments) == 2:
        for nn in range(len(df)):
            if df.loc[nn]["timepoint_"+args.segments[0]] and df.loc[nn]["timepoint_"+args.segments[1]]:
                fname = 'flownet_%s_%d.mp4' % (df.iloc[nn]['meta_video_file_name'][:-4], df.iloc[nn]['meta_position_nn'])
                video_input_fn = os.path.join(args.data_directory, args.segments, "optical_flow", fname)
                print("input: ", video_input_fn)
                if not os.path.exists(video_input_fn):
                    continue
                label = df.iloc[nn][args.label]
                if label == 0:
                    video_output_fn = os.path.join(destination_negative, fname)
                elif label == 1:
                    video_output_fn = os.path.join(destination_positive, fname)
                else:
                    print("skip")
                    continue
                print("output: ", video_output_fn)
                copyfile(video_input_fn, video_output_fn)
    elif len(args.segments) == 1:
        ############################################################
        # generate kinematics here, using start_val and end_val
        ############################################################
        for nn in range(len(df)):
            if df.loc[nn]["timepoint_"+args.segments[0]]:
                timepoint = df.loc[nn]["timepoint_"+args.segments[0]]
                video_input_fn = os.path.join(args.raw_directory, df.iloc[nn]['meta_video_file_name'])
                fname = '%s_%d.mp4' % (df.iloc[nn]['meta_video_file_name'][:-4], df.iloc[nn]['meta_position_nn'])
                print(video_input_fn)
                if not os.path.exists(video_input_fn) or isinstance(timepoint, list):
                    print("FAIL")
                    continue
                # make this adapt to use max padding, finding the min between the set value and hte distance to the neighboring timepoint
                start_val = timepoint - float(args.window)
                end_val = timepoint + float(args.window)
                start_time = '0' + str(datetime.timedelta(seconds=start_val))
                n_frames = round(30.*(end_val-start_val))
                label = df.iloc[nn][args.label]
                if label == 0:
                    video_output_fn = os.path.join(destination_negative, fname)
                elif label == 1:
                    video_output_fn = os.path.join(destination_positive, fname)
                else:
                    print("skip")
                    continue
                cmd = 'ffmpeg -ss %s -i %s -an -vcodec h264 -r 30 -vframes %d %s' % (start_time, video_input_fn, n_frames, video_output_fn)
                os.system(cmd)
        # Generate the flow for destination_positive and destination_negative
        # Flow1 = convert_using_flownet.OpticalFlow(args)
        # Flow1.generate_flow()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input argument
    parser.add_argument("--raw_directory", default = settings.raw_directory, help = "Path to the data directory")
    # parser.add_argument("--data_labels", default = settings.data_labels, help = "Path to labels")
    parser.add_argument("--data_labels", default = settings.data_labels, help = "Path to labels")
    parser.add_argument("--data_directory", default = settings.data_directory, help = "Path to the data directory")
    parser.add_argument("--output_directory", default = settings.label_directory, help = "Path to the output directory")
    #parser.add_argument("--segments", default = "CD", help = "segments")
    parser.add_argument("--segments", default = "B", help = "segments")
    parser.add_argument("--label", default = "label_needle positionB", help = "label")
    parser.add_argument("--window", default = 0.5, help = "window")
    parser.add_argument('--inclusive', dest='inclusive', action='store_true', help = "if inclusive, use first index of start, last index of end for list timepoints")
    args = parser.parse_args()
    splice(args)

