import csv
import os
import pandas as pd

csv_input =  "/mnt/datasets/fvpbignas/datasets/moments-in-time/Moments-in-Time-Raw/training.csv"
output_root = "/mnt/datasets/fvpbignas/datasets/moments-in-time/Moments-in-Time-Aug/training"
data = pd.read_csv(csv_input)

with open("output_csv", mode="w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for d in range(len(data)):
        label = self.data_frame.at[d, "label"]
        file_path = self.data_frame.at[d, "path"]
        video_name = os.path.basename(file_path).replace(".mp4", "")
        root_path = os.path.join(output_root, label, video_name)
        if check_path:
            writer.writerow([label, root_path])


def check_path(file_path):
    if not os.path.exists(file_path):
        return False
    subdirs = glob.glob(file_path + "/*/")
    if len(sub_dirs) < 3:
        return False
    experts = ["location-embeddings", "img-embeddings"]
    for d in subdirs:
        for i in experts:
            subdir = os.path.join(d, i)
            if len(os.listdir(subdir)) < 16:
                return False

    return True
