import glob
import pandas as pd
import ast
import random
import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl
import json
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler

class MITDataModule(pl.LightningDataModule):

    def __init__(self, train_data, val_data, config):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.bs = self.config["batch_size"]
    #     self.label_df = self.load_labels(
    #         "/home/ed/self-supervised-video/data_processing/moments_categories.csv")

    # def load_labels(self, label_root):
    #     label_df = pd.read_csv(label_root)
    #     label_df.set_index('label', inplace=True)
    #     print("len of labels = ", len(label_df))
    #     return label_df

    # def collect_labels(self, label):
    #     index = self.label_df.loc[label]["id"]
    #     return index

    def custom_collater(self, batch):

        return {
            'label': [x['label'] for x in batch],
            'experts': [x['expert_list'] for x in batch],
            'path': [x['path'] for x in batch]
        }

    # def prepare_data(self):
    #    data = self.load_data(self.pickle_file)
    #    self.data = self.clean_data(data)

    def clean_data(self, data_frame, train=False):

        print("cleaning data")
        print(len(data_frame))
        for i in range(len(data_frame)):

            data = data_frame.at[i, "data"]
            # label = data_frame.at[i, "label"]
            # label = self.collect_labels(label)
            # data_frame.at[i, "label"] = label

            drop = False
            # if len(data.values()) > 3:
            #     print(data.values())
            for key, value in data.items():
                if len(value.keys()) < 2:
                    drop = True
                # if train:
                #     if not "img-embeddings" in value.keys():
                #         del data[key]
                # else:
                #     if not "test-img-embeddings" in value.keys():
                #         del data[key]
            if drop:
                print("dropping missing experts")
                data_frame = data_frame.drop(i)
                continue

            data_chunk = list(data.values())

            if len(data_chunk) < 2:
                print("dropping index with no data", i, len(data_chunk))
                data_frame = data_frame.drop(i)
                continue

            # x = [len(data) for data in data_chunk]
            # if sum(x) < len(x) * 3:
            #    print("dropping index with incomplete data", i, len(data))
            #    data_frame = data_frame.drop(i)
            #    continue

                # test = []
                # for f in data[1]: # data1 == img_embeddings, data2 == motion?, data0=location
                #   print(f)
                #   f = torch.load(f)
                #  f = f.squeeze()
                # test.append(f)
                # print(f.dim)
                # if f.dim() > 0:
                #    test.append(f)
                # else:
                #    data_frame = data_frame.drop(i)
                #    continue
                # try:
                #    test = torch.cat(test, dim=-1)
                # except:
                #    data_frame = data_frame.drop(i)
                #    print("dropping", i)
                #    continue
                # print(test.shape[0])
                # if test.shape[0] != 2560:
                #    print("dropping", i)
                #    data_frame = data_frame.drop(i)
                #    continue

        data_frame = data_frame.reset_index(drop=True)
        print(len(data_frame))

        return data_frame

    def load_data(self, db):
        print("loading data")
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

        # TODO remove - 64 Bx2 testing only
        data_frame = data_frame.head(10000)

        return data_frame

    def create_sampler(self, df):
        print("balancing data")
        labels_unique, counts = np.unique(df['label'], return_counts=True)
        sample_weights = [0] * len(df)
        class_list = [0] * 305

        print(f"unique labels :{labels_unique}{counts}")
        # class_weights = [sum(counts) / c for c in counts]
        class_weights = 1./torch.tensor(counts, dtype=torch.float)
        for n, i in enumerate(labels_unique):
            class_list[i] = class_weights[n]

        for n, e in enumerate(df['label']):
            class_weight = class_list[e]
            sample_weights[n] = class_weight.detach()
        sampler = WeightedRandomSampler(
            sample_weights, len(df['label']), replacement=True)
        return sampler

    def setup(self, stage):

        self.train_data = self.load_data(self.train_data)
        self.train_data = self.clean_data(self.train_data, train=True)
        self.weighted_sampler = self.create_sampler(self.train_data)
        self.val_data = self.load_data(self.val_data)
        self.val_data = self.clean_data(self.val_data, train=False)

    def train_dataloader(self):
        print("Loading train dataloader")
        return DataLoader(MITDataset(self.train_data, self.config, train=True), self.bs, sampler=self.weighted_sampler, collate_fn=self.custom_collater, num_workers=0, drop_last=True)

    def val_dataloader(self):
        return DataLoader(MITDataset(self.val_data, self.config, train=False), self.bs, shuffle=False, collate_fn=self.custom_collater, num_workers=0, drop_last=True)
    # For now use validation until proper test split obtained

    def test_dataloader(self):
        return DataLoader(MITDataset(self.train_data, self.config, train=False), 1, shuffle=False, collate_fn=self.custom_collater, num_workers=0)


class MITDataset(Dataset):
    def __init__(self, data, config, train=True):
        super().__init__()

        self.config = config
        self.data_frame = data
        self.train = train
        self.label_df = self.load_labels(
            "/home/ed/self-supervised-video/data_processing/moments_categories.csv")

    def __len__(self):
        return len(self.data_frame)

    def collect_one_hot_labels(self, label):
        label_array = np.zeros(305)
        index = self.label_df.loc[label]["id"]
        label_array[index] = 1
        label_array = torch.LongTensor(label_array)
        print(label_array)
        return label_array

    def collect_labels(self, label):
        index = self.label_df.loc[label]["id"]
        return index

    def load_labels(self, label_root):
        label_df = pd.read_csv(label_root)
        label_df.set_index('label', inplace=True)
        print("len of labels = ", len(label_df))
        return label_df

    def load_tensor(self, tensor):
        tensor = torch.load(tensor, map_location=torch.device('cpu'))
        if tensor.shape[-1] != 2048:
            tensor = nn.ConstantPad1d((0, 2048 - tensor.shape[-1]), 0)(tensor)
        # tensor = torch.load(tensor).detach()
        # tensor = torch.load(tensor, map_location=torch.device('cpu'))
        return tensor

    def __getitem__(self, idx):

        label = self.data_frame.at[idx, "label"]
        # label = self.collect_labels(label)
        data = self.data_frame.at[idx, "data"]
        path = self.data_frame.at[idx, "path"]

        # x_i, x_j = random.sample(list(data.values()), 2)
        expert_list = []
        target_len = 3
        if self.config["cls"]:
            target_len += 1

        if self.config["mixing_method"] == "double_trans":

            for expert in self.config["experts"]:

                expert_t_list = []
                if self.config["cls"]:
                    expert_t_list.append(torch.rand(1, 2048))
                # use test experts
                if not self.train:
                    expert = "test-" + expert

                # not ordered dictionary so need to sort
                temp_list = []
                # if sample is of len 4 remove the last one 
                for i, d in enumerate(data.values()):
                    try:
                        temp_list.append(d[expert][0])
                    except KeyError:
                        continue
                        
                temp_list = sorted(temp_list)
                for i, d in enumerate(temp_list):
                    if i < target_len:
                        expert_t_list.append(self.load_tensor(temp_list[i]))
                while len(expert_t_list) < target_len:
                    expert_t_list.append(expert_t_list[0])

                print(len(expert_t_list))

                assert(len(expert_t_list) == target_len)
                t_tens = torch.stack(expert_t_list)
                t_tens = t_tens.unsqueeze(0)
                expert_list.append(t_tens)
        else:
            # use test experts
            if not self.train:
                expert = "test-" + expert

            # not ordered dictionary so need to sort
            # if sample is of len 4 remove the last one 
            for i, d in enumerate(data.values()):
                try:
                    expert_list.append(d[expert][0])
                except KeyError:
                    continue
        
            temp_list = sorted(temp_list)
            for i, d in enumerate(temp_list):
                if i < 3:
                    expert_list.append(self.load_tensor(temp_list[i]).unsqueeze(0))
            while len(expert_list) < 3:
                expert_list.append(expert_list[0])

            assert(len(expert_list) == 3)

        expert_list = torch.cat(expert_list, dim=0)
        expert_list = expert_list.squeeze()
        expert_list = expert_list.squeeze()
        label = torch.tensor([label])

        # for index, i in enumerate(x_i):
        #    print(i)
        #    t = torch.load(i)
        #    experts_xi.append(t.squeeze())

        # for index, i in enumerate(x_j):
        #    print(i)
        #    t = torch.load(i)
        #    experts_xj.append(t.squeeze())


        return {"label": label, "path": path, "expert_list": expert_list}

