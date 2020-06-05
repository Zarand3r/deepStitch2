#%%
import pandas as pd
import ast
from tqdm import tqdm
import datetime
import os

df = pd.read_excel('/home/fluongo/code/usc_project/USC_lightning/preprocessing/prep_race_videos/RACE score_IT format.xlsx')

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

# Cuts to make A->B, B->C, C->D0, D0-->DN, DN --> E, E--> F, F -->G
# Names: AB, BC, CD, [DD0..DDN] ,DE, EF, FG

# THe way this is written it assume that G will alwatys have only a single value, e.g. GG cannot exist

#cmd = 'ffmpeg -ss %s -i %s -an -vcodec h264 -r 30 -vframes %d %s' % (start_time, video_id, n_frames, video_id_out)
#os.system(cmd)

nn = 0
all_timepoints = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

input_dir = '/home/fluongo/Dropbox/balint_data/caltech_format'
output_dir = '/home/fluongo/Dropbox/balint_data/caltech_format/cuts'

for xx in range(6):
    df['meta_cut_%s%s' % (all_timepoints[xx], all_timepoints[xx+1])] = None


# First do all the nonletter cuts
for nn in tqdm(range(len(df))):
    kk = 0
    while kk < 6:
        start_col = 'timepoint_%s' % all_timepoints[kk]
        end_col = 'timepoint_%s' % all_timepoints[kk+1]
        #print('on %s to %s' % (start_col, end_col))
        start_val, end_val = df.iloc[nn][start_col], df.iloc[nn][end_col]
        if start_val == None or end_val == None:
            #print('ff 2 due to None val')
            kk+=2
            pass
        else:
            # If start_val is a list make it the last one
            if type(start_val) == list:
                start_val = start_val[-1]
            if type(end_val) == list:
                end_val = end_val[0]
            #print(start_val)
            start_time = '0' + str(datetime.timedelta(seconds=start_val))
            n_frames = round(30.*(end_val-start_val))
            video_input_fn = df.iloc[nn]['meta_video_file_name']
            video_input_fn = os.path.join(input_dir, video_input_fn)
            video_output_fn = '%s_%s%s_%02d.mp4' % (df.iloc[nn]['meta_video_file_name'][:-4], all_timepoints[kk], all_timepoints[kk+1], df.iloc[nn]['meta_position_nn'])
            df.at[nn, 'meta_cut_%s%s' % (all_timepoints[kk], all_timepoints[kk+1])] = video_output_fn
            video_output_fn = os.path.join(output_dir, video_output_fn)

            if not os.path.exists(video_input_fn):
                raise ValueError('The file you are trying to chop does not exist')
            else:
                if not os.path.exists(video_output_fn):
                    cmd = 'ffmpeg -ss %s -i %s -an -vcodec h264 -r 30 -vframes %d %s' % (start_time, video_input_fn, n_frames, video_output_fn)
                    #print(cmd)
                    os.system(cmd)
                else:
                    print('already exists so skipping...')
                #print(cmd)
                kk+=1

# Now do the CC and DD versions
nn = 0

input_dir = '/home/fluongo/Dropbox/balint_data/caltech_format'
output_dir = '/home/fluongo/Dropbox/balint_data/caltech_format/cuts'

df['meta_cut_CC'] = None
df['meta_cut_DD'] = None

for nn in range(len(df)):
    for tp_letter in ['C', 'D']:
        curr_entry = df.iloc[nn]['timepoint_%s' % tp_letter]
        if type(curr_entry) == list:
            if len(curr_entry) > 1:
                fns = []
                for cnt, kk in enumerate(range(len(curr_entry)-1)):

                    start_time = '0' + str(datetime.timedelta(seconds=curr_entry[kk]))
                    n_frames = round(30.*(curr_entry[kk+1]-curr_entry[kk]))

                    video_input_fn = df.iloc[nn]['meta_video_file_name']
                    video_input_fn = os.path.join(input_dir, video_input_fn)
                    video_output_fn = '%s_%s%s%d_%02d.mp4' % (df.iloc[nn]['meta_video_file_name'][:-4], tp_letter, tp_letter, cnt, df.iloc[nn]['meta_position_nn'])
                    fns.append(video_output_fn) # Add to the list for later
                    video_output_fn = os.path.join(output_dir, video_output_fn)

                    if not os.path.exists(video_input_fn):
                        raise ValueError('The file you are trying to chop does not exist')
                    else:
                        if not os.path.exists(video_output_fn):
                            cmd = 'ffmpeg -ss %s -i %s -an -vcodec h264 -r 30 -vframes %d %s' % (start_time, video_input_fn, n_frames, video_output_fn)
                            #print(cmd)
                            os.system(cmd)
                        else:
                            print('already exists so skipping...')
                        #print(cmd)
                        kk+=1
                df.at[nn, 'meta_cut_%s%s' % (tp_letter, tp_letter)] = fns

# Add an existing column for first and last for all Cs and Ds
'label_needle_entry_angleC'
'label_hitmiss_timepointC'
'label_hitmissD'
'label_needle_driving_1D'






#df.to_excel('/home/fluongo/code/usc_project/USC_lightning/preprocessing/prep_race_videos/RACE_francisco_export.xlsx')


# %%
from collections import Counter


# Go through and if its a list assign the first one as the label
tps = ['AB', 'BC', 'CD', 'DE', 'EF', 'FG']
lbs = ['needle_positionB', 'needle_entry_angleC', 'hitmiss_timepointC', 'hitmissD', 'needle_driving_1D', 'needle_driving_2FG']

df_count = pd.DataFrame(index = tps, columns = pd.MultiIndex.from_product([tuple(lbs), (0, 1)],
                           names=['name', 'val']))

for a in tps:
    for b in lbs:
        label_name = 'label_%s' %b
        cut_name = 'meta_cut_%s' % a
        df_sub = df[[cut_name, label_name]].dropna()
        # Do the long way of counting
        cnts = []
        for ii in range(len(df_sub)):
            if type(df_sub.iloc[ii][label_name]) != list:
                cnts.append(df_sub.iloc[ii][label_name])
            else:
                cnts.append(df_sub.iloc[ii][label_name][0])
        
        c_cnt = Counter(cnts)
        for k in [0, 1]:
            df_count.at[a, (b, k)] = c_cnt[k]
        





# %%
