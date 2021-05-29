import csv
import os
import pandas as pd
import glob
import tqdm
import multiprocessing as mp

csv_input =  "/mnt/fvpbignas/datasets/moments_in_time/Moments_in_Time_Raw/trainingSet.csv"
output_root = "/mnt/fvpbignas/datasets/moments_in_time/Moments_in_Time_Aug/training"
data_frame = pd.read_csv(csv_input)

def check_path(index):
    file_path = data_frame.at[index, "path"] 
    video_name = os.path.basename(file_path).replace(".mp4", "")
    
    label = data_frame.at[index, "label"]
    root_path = os.path.join(output_root, label, video_name)
    if not os.path.exists(root_path):
        return False
    subdirs = glob.glob(root_path + "/*/")
    if len(subdirs) < 3:
        return False
    # experts = ["location-embeddings", "img-embeddings", "video-embeddings"]
    # for d in subdirs:
    #    for i in experts:
    #        subdir = os.path.join(d, i)
    #        if len(os.listdir(subdir)) < 1:
    #            return False

    return [label, root_path]


def mp_handler():
    p = mp.Pool(40) 
    with open("sanitized.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        indexes = list(range(len(data_frame)))
        for result in p.imap(check_path, tqdm.tqdm(indexes, total=len(indexes))):
            if result:
                writer.writerow(result)


if __name__ == "__main__":
    mp_handler()


