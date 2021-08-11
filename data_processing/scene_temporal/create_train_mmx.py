import csv
import os
import pandas as pd
import glob
import tqdm
import multiprocessing as mp
import numpy as np
import torch
import pickle
import resource


input_dir = "/mnt/bigelow/scratch/mmx_aug/"

def create_embedding_dict(filepath):
    genre_name = filepath.split("/")[-3:-1]
    orig_dir = filepath.replace("/mnt/bigelow/scratch/mmx_aug", "/mnt/fvpbignas/datasets/mmx_raw")
    print(orig_dir)
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
            chunk_dict[os.path.basename(chunk_str)] = expert_list

        scene_str = scene.split("/")[-2]
        scene_dict[os.path.basename(scene_str)] = chunk_dict
    out_dict = {"label": label, "name": name, "scenes": scene_dict}
    return out_dict

def create_scene_dict(filepath):
    experts = ["location-embeddings", "img-embeddings", "video-embeddings", "audio-embeddings"]

    orig_dir = filepath.replace("/mnt/bigelow/scratch/mmx_aug", "/mnt/fvpbignas/datasets/mmx_raw")

    scene = orig_dir.split("/")[-2]
   
    dirs = os.listdir(orig_dir)
    meta_data = os.path.join(orig_dir, "meta.pkl")
    with open(meta_data, "rb") as pickly:
        label = pickle.load(pickly)
    chunk_dict = dict()
    for chunk in glob.glob(filepath + "/*/"):
        expert_list = []
        for expert_dir in experts:
            tens_dir = os.path.join(chunk, expert_dir)
            try:
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
                    # No audio embedding available
                    continue
            except:
                continue
        chunk_str = chunk.split("/")[-2]
        chunk_dict[os.path.basename(chunk_str)] = expert_list
    scene_dict = {"path":orig_dir, "scene":scene, "label":label, "data":chunk_dict}
    return scene_dict


def squish_folders(input_dir):
    all_files = []
    for genres in tqdm.tqdm(glob.glob(input_dir + "/*/")):
        for movies in os.listdir(genres):
            path = os.path.join(genres, movies)
            for scene in glob.glob(path + "/*/"):
                all_files.append(scene)
    print("length of files", len(all_files))
    with open("cache.pkl", "wb") as cache:
        pickle.dump(all_files, cache)


def mp_handler():
    p = mp.Pool(20)
    data_list = []
    count = 0
    with open("cache.pkl", 'rb') as cache:
        working_dirs = pickle.load(cache)
        print(len(working_dirs))
    with open("mmx_tensors.pkl", 'ab') as pkly:
        for result in p.imap(create_scene_dict, tqdm.tqdm(working_dirs, total=len(working_dirs))):
            if result:
                pickle.dump(result, pkly)
            # data_list.append(result)
            # if len(data_list) + count == 50000:
            #     with open("master_tensors1.pkl", "wb") as csv_file:
            #         pickle.dump(data_list, csv_file)
            #         print("dumped 500000")
            #         data_list = []
            #         count = 50001

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    mp_handler()
    # squish_folders(input_dir)
