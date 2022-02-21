import pandas as pd
from simplejson import OrderedDict
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
from random import randint
from collections import OrderedDict
import glob
from sklearn.utils import shuffle

class MMXLightDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, config):
        super().__init__()
        self.csv_path = csv_path
        self.config = config
        self.bs = config["batch_size"]
        self.seq_len = config["seq_len"]

    def prepare_data(self):
        self.data_frame = pd.read_csv(self.csv_path)
        self.data_frame = shuffle(self.data_frame)
        self.data_frame.reset_index(drop=True)
        self.train_data = self.data_frame.iloc[:2000,:]
        self.val_data = self.data_frame.iloc[2000:2500,:]
        self.train_loader = DataLoader(MMXLightDataset(self.train_data, self.config, state="train"), self.bs,  shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
        self.val_loader = DataLoader(MMXLightDataset(self.val_data, self.config, state="val"), self.bs, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
        print("loaded training data", len(self.train_data))
        print("loaded val data", len(self.val_data))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return DataLoader(MMXLightDataset(self.val_data, self.config, state="test"), self.bs, shuffle=False,drop_last=True,  num_workers=5)

class MMXLightDataset(Dataset):
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

        self.train_vid = transforms.Compose([
            transforms.Resize(120),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            transforms.RandomErasing(),
        ])

        self.val_vid = transforms.Compose([
            transforms.Resize(120),
            transforms.CenterCrop(112),
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
 
    def collect_labels(self, label):

        target_names = ['Action', 'Animation', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                        'Fantasy', 'History', 'Horror', 'Music', 'Romance', 'Mystery', 'TVMovie', 'ScienceFiction', 'Thriller',  'War', 'Western']
        label_list = np.zeros(19)
        for i, genre in enumerate(target_names):
            if genre in label:
                label_list[i] = 1
        if np.sum(label_list) == 0:
            label_list[6] = 1

        return label_list

    def vid_trans(self, vid):
        if self.state == "train":
            vid = self.train_vid(vid)
        else:
            vid = self.val_vid(vid)
        return vid

    def __getitem__(self, idx):

        x = torch.empty([self.max_len, 3, 224, 224])
        v = torch.empty([self.max_len, 12, 3, 112, 112])

        img_list = torch.full_like(x, 0)
        vid = torch.full_like(v, 0)

        row = self.data_frame.iloc[idx]
        labels = []
        for i in range(1, 6):
            labels.append(row[f"g{i}"])
        img_root = row["img_root"]

        target = self.collect_labels(labels)
        scenes = sorted(glob.glob(img_root + "/*"))
        frame_dict = OrderedDict()

        for scene, img_dir in enumerate(scenes):
            frame_list = sorted(glob.glob(img_dir + "/*.png"))
            frame_dict[scene] = frame_list
        scene_len = len(scenes)
        i = 0
        for j in range(self.max_len):
            k = 0
            imgs = frame_dict[i]
            img_len = len(imgs)
            for x in range(12):
                vid[i][k] = self.vid_trans(self.pil_loader(imgs[k]))
                k += 1
                k  = k % img_len
            img_list[i] = self.img_trans(self.pil_loader(imgs[3]))
            i += 1
            i = i % scene_len

        return target, img_list, vid


