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
        # data_frame = data_frame.head(100)
        return data_frame

    def setup(self, stage):
        self.train_data = self.load_data(self.train_data)
        self.val_data = self.load_data(self.val_data)

    def train_dataloader(self):
        return DataLoader(MMXFrameDataset(self.train_data, self.config, state="train"), self.bs, shuffle=True, num_workers=5, drop_last=True)

    def val_dataloader(self):
        return DataLoader(MMXFrameDataset(self.val_data, self.config, state="val"), self.bs, shuffle=False, num_workers=5, drop_last=True)

    def test_dataloader(self):
        return DataLoader(MMXFrameDataset(self.val_data, self.config, state="test"), self.bs, shuffle=False, drop_last=True)


class MMXFrameDataset(Dataset):
    def __init__(self, data, config, state="train"):
        super().__init__()

        self.config = config
        self.data_frame = data
        self.seq_len = self.config["seq_len"]
        self.state = state
        
        self.transform = transforms.Compose([transforms.Resize(300), 
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                             std=[0.229, 0.224, 0.225]), 
                                        ])

    def __len__(self):
        return len(self.data_frame)

    def pil_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img_tensor = self.transform(img)
            f.close()
            
        img_tensor = img_tensor.float()
        return img_tensor

    def __getitem__(self, idx):

        # retrieve labels
        label = self.data_frame.at[idx, "label"]
        #label = self.label_tidy(label) -> moved to training loop
        #path = self.data_frame.at[idx, "path"]
        #label = torch.tensor(label).unsqueeze(0)    # Covert label to tensor
        scenes = self.data_frame.at[idx, "scenes"]

        # iterate through the scenes for the trailer
        count = 0
        trailer_list = np.zeros((self.config["seq_len"], self.config["clip_len"], self.config["frame_len"], 3, 224, 224), dtype=float) # create empty array [0, 100]
        for j, s in enumerate(scenes.values()):
            if j < self.config["seq_len"]: 
                for n, c in enumerate(s.values()):
                    if n < self.config["clip_len"]:
                        for f, img in enumerate(c[:self.config["frame_len"]]):
                            img_tensor = self.pil_loader(img)
                            trailer_list[j][n][f] = img_tensor
            
        return label, trailer_list
 