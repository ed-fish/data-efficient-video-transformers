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
import random


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
            'experts': [x['experts'] for x in batch],
            'path': [x["path"] for x in batch]
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
        data_frame = data_frame.head(1000)
        return data_frame

    def setup(self, stage):

        self.train_data = self.load_data(self.train_data)
        self.train_data = self.clean_data(self.train_data)
        self.val_data = self.load_data(self.val_data)
        self.val_data = self.clean_data(self.val_data)

    def train_dataloader(self):
        return DataLoader(MMXDataset(self.train_data, self.config, state="train"), self.bs, shuffle=True, num_workers=15, drop_last=True)

    def val_dataloader(self):
        return DataLoader(MMXDataset(self.val_data, self.config, state="val"), self.bs, shuffle=False, num_workers=15, drop_last=True)

    def test_dataloader(self):
        return DataLoader(MMXDataset(self.val_data, self.config, state="test"), self.bs, shuffle=False, collate_fn=self.custom_collater, drop_last=True)


class MMXDataset(Dataset):
    def __init__(self, data, config, state="train"):
        super().__init__()

        self.config = config
        self.data_frame = data
        self.aggregation = None
        self.seq_len = self.config["seq_len"]
        self.state = state

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
        if self.state == "val":
            expert = "test-" + expert

        try:
            scene_list = path[list(path.keys())[0]][expert]
        except KeyError:
            try:
                scene_list = path[list(path.keys())[0]][ex]
            except KeyError:
                scene_list = False
        except IndexError:
            scene_list = False
        except FileNotFoundError:
            scene_list = False
        return scene_list

    def retrieve_tensors(self, path, expert):
        tensor_paths = self.return_expert_path(path, expert)

        if tensor_paths:
            if expert == "img-embeddings" or expert == "location-embeddings":
                tensor_paths = tensor_paths[-1]
                try:
                    t = self.load_tensor(tensor_paths)
                except FileNotFoundError:
                    t = torch.zeros((1, 2048))
                if expert == "audio-embeddings":
                    t = t.unsqueeze(0)
            if t.shape[-1] != 2048:
                # zero pad dimensions.
                t = nn.ConstantPad1d((0, 2048 - t.shape[-1]), 0)(t)
        else:
            t = torch.zeros((1, 2048))
        if self.state == "train":
            t = self.add_transforms(t)
        return t

    def add_transforms(self, x):
        if random.random() < 0.3:
            x = torch.zeros((1, 2048))
        if random.random() < 0.3:
            x = x + (0.1**0.5)*torch.randn(1, 2048)
        return x

    def label_tidy(self, label):
        if len(label) == 2:
            return self.collect_labels(label[0])
        else:
            return self.collect_labels(label)

    def multi_model_item_collection(self, scene_path):
        expert_tensor_list = []
        for expert in self.config["experts"]:
            if self.config["mixing_method"] == "concat-norm":
                t = F.normalize(self.retrieve_tensors(
                    scene_path, expert), p=2, dim=-1)
            else:
                t = self.retrieve_tensors(scene_path, expert)
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

    def __getitem__(self, idx):

        # retrieve labels
        label = self.data_frame.at[idx, "label"]
        label = self.label_tidy(label)
        path = self.data_frame.at[idx, "path"]
        label = torch.tensor(label).unsqueeze(0)    # Covert label to tensor
        scenes = self.data_frame.at[idx, "scenes"]
        expert_list = []

        # iterate through the scenes for the trailer

        for d in scenes.values():
            if len(expert_list) < self.seq_len:  # collect tensors until sequence length
                expert_tensor_list = []
                # otherwise return one expert
                try:
                    tensor = self.retrieve_tensors(
                        d, self.config["experts"][0])
                except IndexError:
                    continue
                except KeyError:
                    print("key error", d)
                    continue
                except IsADirectoryError:
                    continue
                expert_list.append(tensor)

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

        return {"label": label, "path": path, "experts": expert_list}
