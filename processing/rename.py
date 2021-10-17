import os

# file_dir = "/mnt/md1/richard_bao/balint_data/label_classification_data/negativeBC"
# for f in os.listdir(file_dir):
#     new_f = f.split("_")
#     if new_f[-1][0] == "0":
#         new_f[-1] = new_f[-1][1:]
#     del new_f[-2]
#     new_f = "_".join(new_f)
#     os.rename(os.path.join(file_dir, f), os.path.join(file_dir, new_f))
#     # print(f"renaming {os.path.join(file_dir, f)} to {os.path.join(file_dir, new_f)}")

# file_dir = "/mnt/md1/richard_bao/balint_data/label_classification_data/positiveBC"
# for f in os.listdir(file_dir):
#     new_f = f.split("_")
#     if new_f[-1][0] == "0":
#         new_f[-1] = new_f[-1][1:]
#     del new_f[-2]
#     new_f = "_".join(new_f)
#     os.rename(os.path.join(file_dir, f), os.path.join(file_dir, new_f))


file_dirs = ["/mnt/md1/richard_bao/balint_data/label_classification_data/positiveB", "/mnt/md1/richard_bao/balint_data/label_classification_data/negativeB",
            "/mnt/md1/richard_bao/balint_data/label_classification_data/positiveC", "/mnt/md1/richard_bao/balint_data/label_classification_data/negativeC",
            "/mnt/md1/richard_bao/balint_data/label_classification_data/positiveC1", "/mnt/md1/richard_bao/balint_data/label_classification_data/negativeC1",
            "/mnt/md1/richard_bao/balint_data/label_classification_data/positiveC2", "/mnt/md1/richard_bao/balint_data/label_classification_data/negativeC2"]
for file_dir in file_dirs:
    for f in os.listdir(file_dir):
        new_f = f.split("_")[:-1]
        new_f = "_".join(new_f)+".mp4"
        os.rename(os.path.join(file_dir, f), os.path.join(file_dir, new_f))
        # print(f"renaming {os.path.join(file_dir, f)} to {os.path.join(file_dir, new_f)}")