import csv
import os
import pandas as pd
import glob
import tqdm
import multiprocessing as mp
import numpy as np
import torch
import pickle

input_dir = "/test_data/"

def create_embedding_dict(genre, filepath):

    name = os.path.basename(filepath)
    input_dir = os.path.join("/mnt/bigelow/scratch/mmx_aug", genre, name)
    dirs = os.listdir(input_dir)
    meta_data = os.path.join(input_dir, dirs[0], "meta.pkl")
    label = pickle.load(meta_data)

    # subdirs = [000,001,002] 
    scenes = glob.glob(filepath + "/*/")
    # if len(subdirs) < 2:
    #     return False

    experts = ["location-embeddings", "img-embeddings", "video-embeddings", "audio-embeddings"]
    out_dict = dict()

    for scene in scenes:
        scene_dict = dict()
        # chunks is a list of filepaths [001/001, 001/002]
        chunks = glob.glob(scene, "/*/")
        chunk_dict = dict()
        for chunk in chunks:
            expert_list = []
            for expert_dir in experts:
                tens_dir = os.path.join(chunk, expert_dir)
                if len(os.listdir(tens_dir) > 1):
                    tensor_list = []
                    for tensor in glob.glob(tens_dir + "/*.pt"):
                        tensor_list.append(torch.load(tensor, map_location="cpu"))
                    expert_list.append(tensor_list)
                elif len(os.listdir(tens_dir) == 1):
                    expert_tensor = glob.glob(tens_dir + "/*.pt")
                    expert_tensor = torch.load(expert_tensor[0], map_location="cpu")
                    expert_list.append(expert_tensor)
                else:
                    # No audio embedding available
                    continue

            chunk_dict[os.path.basename(chunk)] = expert_list
        scene_dict[os.path.basename(scene)] = chunk_dict
    out_dict = {"label": label, "name": name, "scenes": scene_dict}

    return out_dict

def squish_folders(input_dir):
    all_files = []
    for directories in glob.glob(input_dir + "/*/"):
        filepaths = glob.glob(directories + "/*/")
        all_files += filepaths
    return all_files


def mp_handler():
    p = mp.Pool(40)
    data_list = []
    count = 0
    working_dirs = squish_folders(input_dir)
    print(len(working_dirs))

    for result in p.imap(create_embedding_dict, tqdm.tqdm(working_dirs, total=len(working_dirs))):
        if result:
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


