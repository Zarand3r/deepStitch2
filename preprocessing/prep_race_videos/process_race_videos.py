#%%
import pandas as pd
import ast

df = pd.read_excel('/home/fluongo/code/usc_project/USC_lightning/preprocessing/prep_race_videos/RACE score_IT format.xlsx')

# Convert the strings to lists and none literals
for ii in range(len(df)):
    for col_name in df.columns:
        if 'label' in col_name or 'timepoint' in col_name:
            curr_entry = df.at[ii, col_name]
            if type(curr_entry) == str:
                print(curr_entry, ii)
                if curr_entry == 'none':
                    df.at[ii, col_name] = ast.literal_eval('None')
                else:
                    df.at[ii, col_name] = ast.literal_eval(curr_entry)

# %%

# Cuts to make A->B, B->C, C->D0, D0-->DN, DN --> E, E--> F, F -->G
# Names: AB, BC, CD, [DD0..DDN] ,DE, EF, FG

# THe way this is written it assume that G will alwatys have only a single value, e.g. GG cannot exist

#cmd = 'ffmpeg -ss %s -i %s -an -vcodec h264 -r 30 -vframes %d %s' % (start_time, video_id, n_frames, video_id_out)
#os.system(cmd)
nn = 0
all_timepoints = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

input_dir = '/home/fluongo/Dropbox/balint_data/caltech_format'
output_dir = '/home/fluongo/Dropbox/balint_data/caltech_format/cuts'

# First do all the nonletter cuts
for nn in range(21, 23):
    kk = 0
    while kk < 6:
        start_col = 'timepoint_%s' % all_timepoints[kk]
        end_col = 'timepoint_%s' % all_timepoints[kk+1]
        print('on %s to %s' % (start_col, end_col))
        start_val, end_val = df.iloc[nn][start_col], df.iloc[nn][end_col]
        if start_val == None or end_val == None:
            print('ff 2 due to None val')
            kk+=2
            pass
        else:
            # If start_val is a list make it the last one
            if type(start_val) == list:
                start_val = start_val[-1]
            if type(end_val) == list:
                end_val = end_val[0]


            print(start_val)
            start_time = '0' + str(datetime.timedelta(seconds=start_val))
            
            n_frames = round(30.*(start_val - end_val))
            
            
            video_input_fn = df.iloc[nn]['meta_video_file_name']
            video_input_fn = os.path.join(input_dir, video_input_fn)
            video_output_fn = '%s_%s%s.mpg' % (df.iloc[nn]['meta_video_file_name'][:-4], all_timepoints[kk], all_timepoints[kk+1])
            video_output_fn = os.path.join(output_dir, video_output_fn)

            cmd = 'ffmpeg -ss %s -i %s -an -vcodec h264 -r 30 -vframes %d %s' % (start_time, video_input_fn, n_frames, video_output_fn)
            print(cmd)
            kk+=1

        print(start_val, end_val)

