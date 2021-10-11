import glob
import pandas as pd
import ast
import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import _pickle as pickle
import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl
import json
from sklearn.model_selection import train_test_split


class MMXDataModule(pl.LightningDataModule):

    def __init__(self, train_data, val_data, config):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.bs = config["batch_size"]
        self.seq_len = config["seq_len"]

    def custom_collater(self, batch):

        return {
            'label': [x['label'] for x in batch],
            'experts': [x['experts'] for x in batch]
        }

    # def prepare_data(self):
    #    data = self.load_data(self.pickle_file)
    #    self.data = self.clean_data(data)

    def clean_data(self, data_frame):
        target_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                        'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Science Fiction', 'Thriller',  'War']

        print("cleaning data")
        print(data_frame.describe())

        longest_seq = 0
        for i in range(len(data_frame)):
            data = data_frame.at[i, "scenes"]
            label = data_frame.at[i, "label"]
            n_labels = 0
            for l in label[0]:
                if l not in target_names:
                    n_labels += 1
            if n_labels == 6:
                data_frame = data_frame.drop(i)
                continue
            data_chunk = list(data.values())
            if len(data_chunk) > longest_seq:
                longest_seq = len(data_chunk)
            if len(data_chunk) < 5:
                data_frame = data_frame.drop(i)
                continue

        data_frame = data_frame.reset_index(drop=True)
        return data_frame

    def load_data(self, db):
        data = []
        with open(db, "rb") as pkly:
            while 1:
                try:
                    # append if data serialised with open file
                    data.append(pickle.load(pkly))
                    # else data not streamed
                    # data = pickle.load(pkly)
                except EOFError:
                    break

        data_frame = pd.DataFrame(data)
        print("data loaded")
        print("length", len(data_frame))
        # data_frame = data_frame.head(1000)
        return data_frame

    def setup(self, stage):

        self.train_data = self.load_data(self.train_data)
        self.train_data = self.clean_data(self.train_data)
        self.val_data = self.load_data(self.val_data)
        self.val_data = self.clean_data(self.val_data)

    def train_dataloader(self):
        return DataLoader(MMXDataset(self.train_data, self.config), self.bs, shuffle=False, collate_fn=self.custom_collater, num_workers=0, drop_last=True)

    def val_dataloader(self):
        return DataLoader(MMXDataset(self.val_data, self.config), self.bs, shuffle=False, collate_fn=self.custom_collater, num_workers=0, drop_last=True)

# For now use validation until proper test split obtained
    def test_dataloader(self):
        return DataLoader(MMXDataset(self.val_data, self.config), self.bs, shuffle=False, collate_fn=self.custom_collater, drop_last=True)


class MMXDataset(Dataset):
    def __init__(self, data, config):
        super().__init__()

        self.config = config
        self.data_frame = data
        self.aggregation = None
        self.seq_len = self.config["seq_len"]

    def __len__(self):
        return len(self.data_frame)

    def collect_labels(self, label):

        target_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                        'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Science Fiction', 'Thriller',  'War']
        label_list = np.zeros(15)

        for i, genre in enumerate(target_names):
            if genre == "Sci-Fi" or genre == "ScienceFiction":
                genre = "Science Fiction"
            if genre in label:
                label_list[i] = 1
        if np.sum(label_list) == 0:
            label_list[5] = 1

        return label_list

    def load_tensor(self, tensor):
        tensor = torch.load(tensor, map_location=torch.device('cpu'))
        return tensor

    def return_expert_path(self, path, expert):
        return path[list(path.keys())[0]][expert]

    def retrieve_tensors(self, path, expert):
        tensor_paths = self.return_expert_path(path, expert)
        if expert == "test-img-embeddings" or expert == "test-location-embeddings":
            tensor_paths = tensor_paths[self.config["frame_id"]]
        if self.config["frame_agg"] == "none":
            t = self.load_tensor(tensor_paths)
            if expert == "audio-embeddings":
                t = t.unsqueeze(0)
        elif self.config["frame_agg"] == "pool":
            pool_list = [self.load_tensor(x) for x in tensor_paths]
            pool_list = torch.stack(pool_list, dim=-1)
            pool_list = pool_list.unsqueeze(0)
            pooled_tensor = F.adaptive_avg_pool2d(
                pool_list, (1, self.config["input_shape"]), dim=-1)
            t = pooled_tensor.squeeze(0)
        if self.config["mixing_method"] == "post_collab":
            if t.shape[-1] != 2048:
                # zero pad dimensions.
                t = nn.ConstantPad1d((0, 2048 - t.shape[-1]), 0)(t)
        return t

    def __getitem__(self, idx):

        # retrieve labels
        label = self.data_frame.at[idx, "label"]
        if len(label) == 2:
            # TODO fix labelling issue - hotfix here
            label = self.collect_labels(label[0])
        else:
            label = self.collect_labels(label)
        label = torch.tensor(label).unsqueeze(0)    # Covert label to tensor
        scenes = self.data_frame.at[idx, "scenes"]
        expert_list = []

        # iterate through the scenes for the trailer

        for i, d in enumerate(scenes.values()):
            try:
                if len(expert_list) < self.seq_len:
                    expert_tensor_list = []
                    # if there are multiple experts find out the mixing method
                    if len(self.config["experts"]) > 1:
                        if self.config["mixing_method"] == "none":
                            assert False, "Mixing method must be defined for multi modal experts"
                        for expert in self.config["experts"]:
                            if self.config["mixing_method"] == "concat-norm":
                                t = F.normalize(self.retrieve_tensors(
                                    d, expert), p=2, dim=-1)
                            else:
                                t = self.retrieve_tensors(d, expert)
                            # Retrieve the tensors for each expert.
                            expert_tensor_list.append(t)
                        if self.config["mixing_method"] == "concat":
                            # concat experts for pre model
                            cat_experts = torch.cat(expert_tensor_list, dim=-1)
                            # expert_list.append(cat_experts)
                            if self.config["cat_norm"] == True:
                                cat_experts = F.normalize(
                                    cat_experts, p=2, dim=-1)
                            if self.config["cat_softmax"] == True:
                                cat_experts = F.softmax(cat_experts, dim=-1)
                            expert_list.append(cat_experts)
                        elif self.config["mixing_method"] == "collab" or self.config["mixing_method"] == "post_collab":
                            expert_list.append(torch.stack(expert_tensor_list))
                    else:
                        # otherwise return one expert
                        expert_list.append(self.retrieve_tensors(
                            d, self.config["experts"][0]))
            except KeyError:
                print("key error")
                # continue
            except IndexError:
                continue
            except IsADirectoryError:
                continue
        if self.config["mixing_method"] == "collab" or self.config["mixing_method"] == "post_collab":
            while len(expert_list) < self.seq_len:
                pad_list = []
                for i in range(len(self.config["experts"])):
                    pad_list.append(torch.zeros_like(expert_list[0][0]))
                expert_list.append(torch.stack(pad_list))
            if self.config["mixing_method"] == "post_collab":
                expert_list = torch.stack(expert_list)
            expert_list = expert_list.squeeze()
        else:
            while len(expert_list) < self.seq_len:
                expert_list.append(torch.zeros_like(expert_list[0]))

            expert_list = torch.cat(expert_list, dim=0)  # scenes
            expert_list = expert_list.unsqueeze(0)

        return {"label": label, "experts": expert_list}

    # for each scene retrieve all the embeddings
    # config["expert"] needs to change to a list so we can mix and match.

    # for expert in experts: cat (load expert 1, loade expert .., n)
    # check same dimension with assert
    # add concat experts to expert_list

    # for i, d in enumerate(scenes.values()):
    #     if len(expert_list) < self.seq_len:
    #         # TODO reformat this code - only good for testing - or even better remove trailing "/" from data pre-processing.
    #         try:
    #             tensor_paths = d[list(d.keys())[0]][self.config["expert"]]
    #             if self.config["frame_agg"] == "none":
    #                 tensor_path = tensor_paths[frame]
    #                 if len(tensor_path) == 1:
    #                     tensor_path = d[list(d.keys())[0]][self.config["expert"]]
    #                     t = self.load_tensor(tensor_path)
    #                     if self.config["expert"] == "audio-embeddings":
    #                         t = t.unsqueeze(0)
    #                 expert_list.append(t)
    #             elif self.config["frame_agg"] == "pool":
    #                 pool_list = [self.load_tensor(x) for x in tensor_paths]
    #                 pool_list = torch.stack(pool_list, dim=-1)
    #                 pool_list = pool_list.unsqueeze(0)
    #                 pooled_tensor = F.adaptive_avg_pool2d(pool_list, (1, self.config["input_shape"]), dim=-1)
    #                 pooled_tensor = pooled_tensor.squeeze(0)
    #                 expert_list.append(pooled_tensor)

    #         except KeyError:
    #             continue
    #         except IndexError:
    #             continue
    #         except IsADirectoryError:
    #             continue

    # while len(expert_list) < self.seq_len:
    #     expert_list.append(torch.zeros_like(expert_list[0]))

    # expert_list = torch.cat(expert_list, dim=0)
    # expert_list = expert_list.unsqueeze(0)

    # return {"label":label, "experts":expert_list}
