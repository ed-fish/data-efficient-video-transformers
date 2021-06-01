import csv
import os
import pandas as pd
import glob
import tqdm
import multiprocessing as mp
import numpy as np
import random
import torch
import pickle

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

    experts = ["location-embeddings", "img-embeddings", "video-embeddings"]
    tensor_list = []
    x_i = subdirs.pop(random.randrange(len(subdirs)))
    x_j = subdirs.pop(random.randrange(len(subdirs)))
    tensor_list_x_i = []
    tensor_list_x_j = []
    for expert_dir in experts:
        subdir_x = os.path.join(x_i, expert_dir)
        subdir_y = os.path.join(x_j, expert_dir)
        x_dir = glob.glob(subdir_x + "/*.pt")
        y_dir = glob.glob(subdir_y + "/*.pt")
        if len(x_dir) > 0 and len(y_dir) > 0:
            expert_tensors_x = torch.load(x_dir[0], map_location='cpu')
            expert_tensors_y = torch.load(y_dir[0], map_location='cpu')
            expert_tensors_x = expert_tensors_x.detach().tolist()
            expert_tensors_y = expert_tensors_y.detach().tolist()
            tensor_list_x_i.append(expert_tensors_x)
            tensor_list_x_j.append(expert_tensors_y)

    return [label, root_path, tensor_list_x_i, tensor_list_x_j]


def mp_handler():
    p = mp.Pool(40) 
    data_list = []
    count = 0
    indexes = list(range(len(data_frame)))
    for result in p.imap(check_path, tqdm.tqdm(indexes, total=len(indexes))):
        if result:
            data_list.append(result)
            if len(data_list) + count == 50000:
                with open("master_tensors1.pkl", "wb") as csv_file:
                    pickle.dump(data_list, csv_file)
                    print("dumped 500000")
                    data_list = []
                    count = 50001
            
            if len(data_list) + count  == 100000:
                with open("master_tensors2.pkl", "wb") as csv_file:
                    pickle.dump(data_list, csv_file)
                    print("dumped 200000")
                    data_list = []
                    count = 100001


            if len(data_list) + count == 150000:
                with open("master_tensors3.pkl", "wb") as csv_file:
                    pickle.dump(data_list, csv_file)
                    print("dumped 300000")
                    data_list = []
                    count = 150001


            if len(data_list) + count == 200000:
                with open("master_tensors2.pkl", "wb") as csv_file:
                    pickle.dump(data_list, csv_file)
                    print("dumped 400000")
                    data_list = []
                    count = 200000


            if len(data_list) + count == 300000:
                with open("master_tensors2.pkl", "wb") as csv_file:
                    pickle.dump(data_list, csv_file)
                    print("dumped 500000")
                    data_list = []
                    count = 300000


            if len(data_list) + count == 400000:
                with open("master_tensors2.pkl", "wb") as csv_file:
                    pickle.dump(data_list, csv_file)
                    print("dumped 600000")
                    data_list = []
                    count = 400000


            if len(data_list) + count == 500000:
                with open("master_tensors2.pkl", "wb") as csv_file:
                    pickle.dump(data_list, csv_file)
                    print("dumped 500000")
                    data_list = []
                    count = 500000


            if len(data_list) + count == 600000:
                with open("master_tensors2.pkl", "wb") as csv_file:
                    pickle.dump(data_list, csv_file)
                    print("dumped 600000")
                    data_list = []
                    count = 600000

if __name__ == "__main__":
    mp_handler()


