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
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import wandb
from torchvision.utils import make_grid
from torch.utils.data.dataloader import default_collate


class MMXFrameDataModule(pl.LightningDataModule):

    def __init__(self, train_data, val_data, config):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.bs = config["batch_size"]
        self.seq_len = config["seq_len"]

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
        data_frame = data_frame.reset_index(drop=True)
        print("length of data", len(data_frame))
        #data_frame = data_frame.head(1000)
        return data_frame

    def setup(self, stage):
        self.train_data = self.load_data(self.train_data)
        self.val_data = self.load_data(self.val_data)

    def train_dataloader(self):
        return DataLoader(MMXFrameDataset(self.train_data, self.config, state="train"), self.bs,  shuffle=True, num_workers=15, drop_last=True)

    def val_dataloader(self):
        return DataLoader(MMXFrameDataset(self.val_data, self.config, state="val"), self.bs, shuffle=False, num_workers=15, drop_last=True)

    def test_dataloader(self):
        return DataLoader(MMXFrameDataset(self.val_data, self.config, state="test"), self.bs, shuffle=False, drop_last=True)


class MMXFrameDataset(Dataset):
    def __init__(self, data, config, state="train"):
        super().__init__()

        self.config = config
        self.data_frame = data
        self.seq_len = self.config["seq_len"]
        self.state = state
        self.max_len = self.config["seq_len"]

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.transform_vid = transforms.Compose([
            transforms.Resize(200),
            transforms.RandomResizedCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])

    def __len__(self):
        return len(self.data_frame)

    def pil_loader(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        return img

    def img_trans(self, img):
        if self.state == "train":
            img = self.train_transform(img)
        else:
            img = self.val_transform(img)
        #img_tensor = img_tensor.float()
        return img

    def __getitem__(self, idx):

        label = self.data_frame.at[idx, "label"]
        scenes = self.data_frame.at[idx, "scenes"]
        x = torch.empty([self.max_len, 3, 224, 224])
        vid = torch.empty([self.max_len, 10, 3, 112, 112])
        img_list = torch.full_like(x, 0)

        # iterate through the scenes for the trailer
        # trailer_list = np.zeros((self.config["seq_len"], self.config["clip_len"], self.config["frame_len"], 3, 224, 224), dtype=float) # create empty array [0, 100]

        num_collected = 0
        for j, s in enumerate(scenes.values()):
            if num_collected == self.config["seq_len"]:
                break
            try:
                clip = s[0]
            except KeyError:
                try:
                    clip = s["000"]
                except KeyError:
                    try:
                        clip = s["0"]
                    except:
                        continue

            for i in range(10):
                vid[num_collected][i] = self.transform_vid(
                    self.pil_loader(clip[i]))
            #img_t = self.img_trans(self.pil_loader(clip[0]))
            #img_list[num_collected] = img_t
            #img_list = []
            num_collected += 1
        # vid = vid.permute(0, 2, 1, 3, 4)
        return label, img_list, vid
