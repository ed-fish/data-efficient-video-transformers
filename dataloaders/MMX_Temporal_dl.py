import glob
import pandas as pd
import ast
import random
import csv
import torch
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

    def __init__(self, train_data,val_data, config):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.bs = self.config["batch_size"]
        self.seq_len = self.config["seq_len"]


    def custom_collater(self, batch):

        return {
                'label':[x['label'] for x in batch],
                'experts':[x['experts'] for x in batch]
                }

    # def prepare_data(self):
    #    data = self.load_data(self.pickle_file)
    #    self.data = self.clean_data(data)

    def clean_data(self, data_frame):
        target_names = ['Action'  ,'Adventure'  ,'Comedy'  ,'Crime'  ,'Documentary'  ,'Drama'  ,'Family' , 'Fantasy'  ,'History'  ,'Horror'  ,'Music' , 'Mystery'  ,'Science Fiction' , 'Thriller',  'War']


        print("cleaning data")
        print(len(data_frame))

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
        print(len(data_frame))
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
        #data_frame = data_frame.head(64)
        return data_frame

    def setup(self, stage):

        self.train_data = self.load_data(self.train_data)
        self.train_data = self.clean_data(self.train_data)
        self.val_data = self.load_data(self.val_data)
        self.val_data = self.clean_data(self.val_data)

    def train_dataloader(self):
        return DataLoader(MMXDataset(self.train_data, self.config), self.bs, shuffle=True, collate_fn=self.custom_collater, num_workers=0, drop_last=True)

    def val_dataloader(self):
        return DataLoader(MMXDataset(self.val_data, self.config), self.bs, shuffle=False, collate_fn=self.custom_collater, num_workers=0, drop_last=True)

# For now use validation until proper test split obtained
    def test_dataloader(self):
        return DataLoader(MMXDataset(self.train_data, self.config), 1, shuffle=False, collate_fn=self.custom_collater, num_workers=0)


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

        target_names = ['Action'  ,'Adventure'  ,'Comedy'  ,'Crime'  ,'Documentary'  ,'Drama'  ,'Family' , 'Fantasy'  ,'History'  ,'Horror'  ,'Music' , 'Mystery'  ,'Science Fiction' , 'Thriller',  'War']
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

    def __getitem__(self, idx):
        label =  self.data_frame.at[idx, "label"]
        if len(label) == 2:
            label = self.collect_labels(label[0])
        else:
            label = self.collect_labels(label)
        label = torch.tensor(label).unsqueeze(0)
        scenes = self.data_frame.at[idx, "scenes"]
        expert_list = []
        frame = 0

        for i, d in enumerate(scenes.values()):
            if len(expert_list) < self.seq_len:
                # TODO reformat this code - only good for testing - or even better remove trailing "/" from data pre-processing.
                try:
                    tensor_path = d[list(d.keys())[0]][self.config["expert"]][frame]
                    if len(tensor_path) == 1:
                        tensor_path = d[list(d.keys())[0]][self.config["expert"]]
                    expert_list.append(self.load_tensor(tensor_path))
                except KeyError:
                    print("key error")
                    continue
                except IndexError:
                    print("index error")
                    continue

        while len(expert_list) < self.seq_len:
            expert_list.append(torch.zeros_like(expert_list[0]))

        expert_list = torch.cat(expert_list, dim=0)
        expert_list = expert_list.unsqueeze(0)

        return {"label":label, "experts":expert_list}


