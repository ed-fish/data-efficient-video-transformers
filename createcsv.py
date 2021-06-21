import csv
import os
import pandas as pd
import glob
import tqdm
import multiprocessing as mp
import numpy as np
import torch
import pickle

input_dir = "/mnt/bigelow/scratch/mmx_aug/"

def create_embedding_dict(filepath):
    genre_name = filepath.split("/")[-3:-1]
    orig_dir = os.path.join("/mnt/fvpbignas/datasets/mmx_raw/", genre_name[0], genre_name[1])
    dirs = os.listdir(orig_dir)
    meta_data = os.path.join(orig_dir, dirs[0], "meta.pkl")
    with open(meta_data, "rb") as pickly:
        label = pickle.load(pickly)

    # subdirs = [000,001,002] 
    scenes = glob.glob(filepath + "/*/")
   
    # if len(subdirs) < 2:
    #     return False

    experts = ["location-embeddings", "img-embeddings", "video-embeddings", "audio-embeddings"]
    out_dict = dict()
    scene_dict = dict()

    for scene in scenes:
        # chunks is a list of filepaths [001/001, 001/002]
        chunks = glob.glob(scene + "/*/")
        chunk_dict = dict()
       
        for chunk in chunks:
            expert_list = []
            for expert_dir in experts:
                tens_dir = os.path.join(chunk, expert_dir)
                if len(os.listdir(tens_dir)) > 1:
                    tensor_list = []
                    for tensor in glob.glob(tens_dir + "/*.pt"):
                        tensor_list.append(torch.load(tensor, map_location="cpu"))
                    expert_list.append(tensor_list)
                elif len(os.listdir(tens_dir)) == 1:
                    expert_tensor = glob.glob(tens_dir + "/*.pt")
                    expert_tensor = torch.load(expert_tensor[0], map_location="cpu")
                    expert_list.append(expert_tensor)
                else:
                    print("no audio")
                    # No audio embedding available
                    continue
            chunk_str = chunk.split("/")[-2]
            chunk_dict[chunk_str] = expert_list
        scene_str = scene.split("/")[-2]
        scene_dict[scene_str] = chunk_dict
    out_dict = {"label": label, "name": genre_name[1], "scenes": scene_dict}

    return out_dict

def squish_folders(input_dir):
    all_files = []
    for directories in glob.glob(input_dir + "/*/"):
        filepaths = glob.glob(directories + "/*/")
        all_files += filepaths
    return all_files


def mp_handler():
    p = mp.Pool(5)
    data_list = []
    count = 0
    working_dirs = squish_folders(input_dir)
    print(len(working_dirs))

    for result in p.imap(create_embedding_dict, tqdm.tqdm(working_dirs, total=len(working_dirs))):
        if result:
            print("completed one")
            print(result)
            # data_list.append(result)
            # if len(data_list) + count == 50000:
            #     with open("master_tensors1.pkl", "wb") as csv_file:
            #         pickle.dump(data_list, csv_file)
            #         print("dumped 500000")
            #         data_list = []
            #         count = 50001

if __name__ == "__main__":
    mp_handler()


