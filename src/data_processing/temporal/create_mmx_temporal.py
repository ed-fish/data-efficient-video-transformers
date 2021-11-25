import csv
import random
import os
import pandas as pd
import glob
import tqdm
import multiprocessing as mp
import numpy as np
import torch
import pickle
import re
import resource

from collections import OrderedDict


input_dir = "/mnt/bigelow/scratch/mmx_aug/"


def create_embedding_dict(filepath):
    genre_name = filepath.split("/")[-3:-1]
    orig_dir = filepath.replace(
        "/mnt/bigelow/scratch/mmx_aug", "/mnt/fvpbignas/datasets/mmx_raw")
   
    dirs = os.listdir(orig_dir)
    meta_data = os.path.join(orig_dir, dirs[0], "meta.pkl")
    with open(meta_data, "rb") as pickly:
        label = pickle.load(pickly)

    # subdirs = [000,001,002]
    scenes = glob.glob(filepath + "/*/")
    scenes = sorted(scenes, key=lambda x: (
        int(re.findall("[0-9]+", x.split("/")[-2])[0])))
    # if len(subdirs) < 2:
    #     return False

    experts = ["img-embeddings", "location-embeddings", "motion-embeddings",
               "test-location-embeddings", "test-img-embeddings", "test-video-embeddings", "audio-embeddings"]
    out_dict = OrderedDict()
    scene_dict = OrderedDict()

    for scene in scenes:
        # chunks is a list of filepaths [001/001, 001/002]
        chunks = glob.glob(scene + "/*/")
        chunk_dict = OrderedDict()

        for chunk in chunks:
            #expert_list = []
            expert_dict = OrderedDict()
            for expert_dir in experts:
                tens_dir = os.path.join(chunk, expert_dir)
                try:
                    if len(os.listdir(tens_dir)) > 1:
                        tensor_list = []
                        for tensor in glob.glob(tens_dir + "/*.pt"):
                            #t = torch.load(tensor, map_location="cpu")
                            #t = t.cpu().detach().numpy()
                            tensor_list.append(tensor)
                        expert_dict[expert_dir] = tensor_list
                        # expert_list.append(tensor_list)
                    elif len(os.listdir(tens_dir)) == 1:
                       
                        expert_tensor = glob.glob(tens_dir + "/*.pt")
                        #expert_tensor = torch.load(expert_tensor[0], map_location="cpu")
                        #expert_tensor = expert_tensor.cpu().detach().numpy()
                        # expert_list.append(expert_tensor[0])
                        expert_dict[expert_dir] = expert_tensor[0]
                    else:
                        # print("no audio")
                        # No audio embedding available
                        continue
                except FileNotFoundError:
                    continue
            chunk_str = chunk.split("/")[-2]
            # chunk_dict[os.path.basename(chunk_str)] = expert_list
            chunk_dict[os.path.basename(chunk_str)] = expert_dict

        scene_str = scene.split("/")[-2]
        scene_dict[os.path.basename(scene_str)] = chunk_dict
    out_dict = {"label": label, "path": orig_dir, "scenes": scene_dict}
    return out_dict


def create_scene_dict_train(filepath):
    experts = ["location-embeddings", "img-embeddings", "video-embeddings"]

    orig_dir = filepath.replace(
        "/mnt/bigelow/scratch/mmx_aug", "/mnt/fvpbignas/datasets/mmx_raw")

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
                        # tensor_list.append(torch.load(tensor, map_location="cpu"))
                        tensor_list.append(tensor)
                    expert_list.append(tensor_list)
                elif len(os.listdir(tens_dir)) == 1:
                    expert_tensor = glob.glob(tens_dir + "/*.pt")
                    #expert_tensor = torch.load(expert_tensor[0], map_location="cpu")
                    expert_list.append(expert_tensor[0])
                else:
                    # No audio embedding available
                    continue
            except:
                continue
        chunk_str = chunk.split("/")[-2]
        chunk_dict[os.path.basename(chunk_str)] = expert_list
    scene_dict = {"path": orig_dir, "scene": scene,
                  "label": label, "data": chunk_dict}
    return scene_dict


def create_scene_dict_test(filepath):
    experts = ["test-location-embeddings",
               "test-img-embeddings", "test-video-embeddings"]

    orig_dir = filepath.replace(
        "/mnt/bigelow/scratch/mmx_aug", "/mnt/fvpbignas/datasets/mmx_raw")

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
                        # tensor_list.append(torch.load(tensor, map_location="cpu"))
                        tensor_list.append(tensor)
                    expert_list.append(tensor_list)
                elif len(os.listdir(tens_dir)) == 1:
                    expert_tensor = glob.glob(tens_dir + "/*.pt")
                    #expert_tensor = torch.load(expert_tensor[0], map_location="cpu")
                    expert_list.append(expert_tensor[0])
                else:
                    # No audio embedding available
                    continue
            except:
                continue
        chunk_str = chunk.split("/")[-2]
        chunk_dict[os.path.basename(chunk_str)] = expert_list
    scene_dict = {"path": orig_dir, "scene": scene,
                  "label": label, "data": chunk_dict}
    return scene_dict


def squish_folders(input_dir):
    all_files = []
    for genres in tqdm.tqdm(glob.glob(input_dir + "/*/")):
        for movies in os.listdir(genres):
            path = os.path.join(genres, movies)
            # use movies rather than individual scenes
            all_files.append(path)
    print("length of files", len(all_files))
    with open("cache.pkl", "wb") as cache:
        pickle.dump(all_files, cache)


def mp_handler():
    p = mp.Pool(30)
    data_list = []
    count = 0
    with open("cache.pkl", 'rb') as cache:
        data = pickle.load(cache)
        random.shuffle(data)
        # Remaining 80% to training set
        train_data = data[:int((len(data)+1)*.90)]
        # Splits 20% data to test set
        test_data = data[int((len(data)+1)*.90):]
        print("training_data", len(train_data))
        print("testing_data", len(test_data))
    big_list = []

    # append to pkl rather than write to pkl

    # with open("mmx_tensors_train.pkl", 'ab') as pkly:
    #     for result in p.imap(create_embedding_dict, tqdm.tqdm(train_data, total=len(train_data))):
    #         if result:
    #             pickle.dump(result, pkly)

    with open("mmx_train_temporal.pkl", 'ab') as pkly:
        for result in p.imap(create_embedding_dict, tqdm.tqdm(train_data, total=len(train_data))):
            if result:
                pickle.dump(result, pkly)

    with open("mmx_val_temporal.pkl", 'ab') as pkly:
        for result in p.imap(create_embedding_dict, tqdm.tqdm(test_data, total=len(test_data))):
            if result:
                pickle.dump(result, pkly)

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    squish_folders(input_dir)
    mp_handler()
