import os
import ast
import datetime
import argparse
import numpy as np
import pandas as pd
from shutil import copyfile
from collections import namedtuple

# Custom imports
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings

from convert_using_flownet import OpticalFlow

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
            if col_name == "label_needle_driving_1_hand":
                continue
            if 'label' in col_name or 'timepoint' in col_name:
                curr_entry = df.at[ii, col_name]
                if type(curr_entry) == str:
                    #print(curr_entry, ii)
                    if curr_entry == 'none':
                        df.at[ii, col_name] = ast.literal_eval('None')
                    else:
                        df.at[ii, col_name] = ast.literal_eval(curr_entry)
                if col_name == args.label:
                    break

    destination_positive = os.path.join(args.output_directory, "positive"+args.segments)
    destination_negative = os.path.join(args.output_directory, "negative"+args.segments)
    if not os.path.exists(destination_positive):
        os.makedirs(destination_positive)

    if not os.path.exists(destination_negative):
        os.makedirs(destination_negative)

    if len(args.segments) == 2:
        for nn in range(len(df)):
            if not (pd.isna(df.loc[nn]["timepoint_"+args.segments[0]]) or pd.isna(df.loc[nn]["timepoint_"+args.segments[1]])):
                fname = 'flownet_%s.mp4' % (df.iloc[nn]['meta_video_file_name'][:-4])
                video_input_fn = os.path.join(args.data_directory, args.segments, "optical_flow", fname)
                print("input: ", video_input_fn)
                if not os.path.exists(video_input_fn):
                    continue
                label = df.iloc[nn][args.label]
                print(label)
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
        for nn in range(len(df)):
            if df.loc[nn]["timepoint_"+args.segments[0]]:
                timepoints = []
                time = df.loc[nn]["timepoint_"+args.segments[0]]
                if isinstance(time, list):
                    timepoints.extend(time)
                else:
                    timepoints.append(time)
                scores = []
                label = df.iloc[nn][args.label]
                if isinstance(label, list):
                    scores.extend(label)
                else:
                    scores.append(label)
                if len(timepoints) != len(scores):
                    print("timepoints and scores are mismatched")
                    continue
                
                count = 1
                for i, timepoint in enumerate(timepoints):
                    score = scores[i]
                    # make this adapt to use max padding, finding the min between the set value and the distance to the neighboring timepoint
                    start_val = timepoint - float(args.window)
                    end_val = timepoint + float(args.window)

                    # generate kinematics here
                    ############################################################
                    kinematics_file = df.iloc[nn]["meta_raw_kinematic_data_name"]
                    if kinematics_file is None or not os.path.exists(kinematics_file):
                        print(kinematics_file)
                        continue
                    kinematics = pd.read_csv(kinematics_file)
                    header = ','.join(list(kinematics.columns))
                    kinematics = kinematics.values
                    start_index = round(29.97*start_val)
                    end_index = round(29.97*end_val)
                    kinematics_splice = kinematics[start_index: end_index+1]
                    output_subdir = os.path.join(settings.kinematics_directory, args.segments[0])
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)
                    kname = f"{os.path.basename(kinematics_file)[:-4]}_{count}.csv"
                    np.savetxt(os.path.join(output_subdir, kname), kinematics_splice, delimiter=",", header=header, comments='')
                    print(f"saving {kinematics_file} to {os.path.join(output_subdir, kname)}")

                    # generate videos here
                    ###########################################################
                    video_input_fn = df.iloc[nn]['meta_video_file_name'].strip()
                    fname = f"{os.path.basename(df.iloc[nn]['meta_video_file_name'])[:-4]}_{count}.mp4"
                    if " " in video_input_fn:
                        video_input_fn = "'"+video_input_fn+"'"
                        fname = "'"+fname+"'"
                    video_input_fn = os.path.join(args.raw_directory, video_input_fn)
                    if isinstance(timepoint, list):
                        print(f"timepoints: {timepoint}")
                        print("Timepoint cannot be a list!")
                        continue
                    start_time = '0' + str(datetime.timedelta(seconds=start_val))
                    n_frames = round(30.*(end_val-start_val))
                    if score == 0:
                        video_output_fn = os.path.join(destination_negative, fname)
                    elif score == 1:
                        video_output_fn = os.path.join(destination_positive, fname)
                    else:
                        print("skip")
                        continue
                    count += 1
                    cmd = 'ffmpeg -ss %s -i %s -an -vcodec h264 -r 30 -vframes %d %s' % (start_time, video_input_fn, n_frames, video_output_fn)
                    os.system(cmd)

        # TODO Generate the flow for destination_positive and destination_negative
        flow_args_positive = {"input_dir": destination_positive, "gpu_id": 0, "window": 4, "join": 2, "div_flow": 2, "max_flow": 20}
        flow_args_positive = namedtuple("args", flow_args_positive.keys())(*flow_args_positive.values())
        Flow_positive = OpticalFlow(flow_args_positive)
        Flow_positive.generate_flow()

        flow_args_negative = {"input_dir": destination_negative, "gpu_id": 0, "window": 4, "join": 2, "div_flow": 2, "max_flow": 20}
        flow_args_negative = namedtuple("args", flow_args_negative.keys())(*flow_args_negative.values())
        Flow_negative = OpticalFlow(flow_args_negative)
        Flow_negative.generate_flow()

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
    parser.add_argument("--window", default = 1, help = "window")
    parser.add_argument('--inclusive', dest='inclusive', action='store_true', help = "if inclusive, use first index of start, last index of end for list timepoints")
    args = parser.parse_args()
    splice(args)

