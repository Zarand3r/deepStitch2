import glob
# Custom imports
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings

# video_files_m4v = glob.glob(f"{settings.TCN_data}/*.m4v")
# video_files_mp4 = glob.glob(f"{settings.TCN_data}/*.mp4")

video_files_m4v = glob.glob(f"{settings.raw_directory}/*.m4v")
video_files_mp4 = glob.glob(f"{settings.raw_directory}/*.mp4")

print(len(video_files_m4v))
print(len(video_files_mp4))

video_files = video_files_m4v + video_files_mp4

with open('videos.txt', 'w') as f:
    for vid in video_files:
        line = vid + " 0 0\n"
        f.write(line)

