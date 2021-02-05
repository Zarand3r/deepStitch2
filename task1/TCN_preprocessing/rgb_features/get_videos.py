import glob

video_files_m4v = glob.glob("/mnt/md1/richard_bao/balint_data/TCN_data/*.m4v")
video_files_mp4 = glob.glob("/mnt/md1/richard_bao/balint_data/TCN_data/*.mp4")

print(len(video_files_m4v))
print(len(video_files_mp4))

video_files = video_files_m4v + video_files_mp4

with open('videos.txt', 'w') as f:
    for vid in video_files:
        line = vid + " 0 0\n"
        f.write(line)

