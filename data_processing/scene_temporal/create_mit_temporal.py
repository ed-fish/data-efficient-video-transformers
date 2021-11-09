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
import resource


def load_labels(label_root):
    label_df = pd.read_csv(label_root)
    label_df.set_index('label', inplace=True)
    print("len of labels = ", len(label_df))
    return label_df


def collect_labels(label_df, label):
    index = label_df.loc[label]["id"]
    return index


def create_dictionary(filepath):
    experts = ["audio-embeddings", "location-embeddings",
               "img-embeddings", "video-embeddings", "test-location-embeddings", "test-img-embeddings", "test-video-embeddings"]

    #orig_dir = filepath.replace("/mnt/bigelow/scratch/mmx_aug", "/mnt/fvpbignas/datasets/mmx_raw")

    label = filepath.split("/")[-3]
    label = collect_labels(label_df, label)

    #dirs = os.listdir(orig_dir)
    #meta_data = os.path.join(orig_dir, "meta.pkl")
    # with open(meta_data, "rb") as pickly:
    #label = os.path.basename(filepath)
    chunk_dict = dict()
    for chunk in glob.glob(filepath + "/*/"):
        expert_dict = {}
        for expert_dir in experts:
            tens_dir = os.path.join(chunk, expert_dir)
            try:
                if len(os.listdir(tens_dir)) > 1:
                    tensor_list = []
                    for tensor in glob.glob(tens_dir + "/*.pt"):
                        # tensor_list.append(torch.load(tensor, map_location="cpu"))
                        tensor_list.append(tensor)
                    expert_dict[expert_dir] = tensor_list
                elif len(os.listdir(tens_dir)) == 1:
                    expert_tensor = glob.glob(tens_dir + "/*.pt")
                    #expert_tensor = torch.load(expert_tensor[0], map_location="cpu")
                    expert_dict[expert_dir] = expert_tensor
                    # expert_list.append(expert_tensor[0])
                else:
                    # No audio embedding available
                    continue
            except:
                continue
        chunk_str = chunk.split("/")[-2]
        chunk_dict[os.path.basename(chunk_str)] = expert_dict
    master_dict = {"path": filepath, "label": label, "data": chunk_dict}
    return master_dict


def squish_folders(input_dir):
    all_files = []
    for labels in glob.glob(input_dir + "/*/"):
        for video in glob.glob(labels + "/*/"):
            all_files.append(video)
        print(labels, "complete")
    print("length of files", len(all_files))
    with open("MIT_validation_cache.pkl", "wb") as cache:
        pickle.dump(all_files, cache)


def mp_handler():
    p = mp.Pool(40)

    squish_folders("/mnt/bigelow/scratch/mit_no_crop/validation")
    with open("MIT_validation_cache.pkl", 'rb') as cache:
        data = pickle.load(cache)
        # random.shuffle(data)

    with open("MIT_validation_temporal.pkl", 'ab') as pkly:
        for result in p.imap(create_dictionary, tqdm.tqdm(data, total=len(data))):
            if result:
                pickle.dump(result, pkly)


if __name__ == "__main__":

    label_df = load_labels(
        "/home/ed/self-supervised-video/data_processing/moments_categories.csv")
    torch.multiprocessing.set_sharing_strategy('file_system')
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    mp_handler()
    # squish_folders(input_dir)
