import pandas as pd
import torch
import _pickle as pickle
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
from random import randint


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
        # data_frame = data_frame.head(2000)
        print("length of data", len(data_frame))
        return data_frame

    def setup(self, stage):
        self.train_data = self.load_data(self.train_data)
        self.val_data = self.load_data(self.val_data)

    def train_dataloader(self):
        return DataLoader(MMXFrameDataset(self.train_data, self.config, state="train"), self.bs,  shuffle=True, num_workers=1, drop_last=True)

    def val_dataloader(self):
        return DataLoader(MMXFrameDataset(self.val_data, self.config, state="val"), self.bs, shuffle=False, num_workers=1, drop_last=True)

    def test_dataloader(self):
        return DataLoader(MMXFrameDataset(self.val_data, self.config, state="test"), self.bs, shuffle=False,drop_last=True,  num_workers=5)


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

    def vid_trans(self, vid):
        if self.state == "train":
            vid = self.train_vid(vid)
        else:
            vid = self.val_vid(vid)
        return vid

    def __getitem__(self, idx):

        label = self.data_frame.at[idx, "label"]
        scenes = self.data_frame.at[idx, "scenes"]
        x = torch.empty([self.max_len, 3, 224, 224])
        v = torch.empty([self.max_len, 12, 3, 112, 112])
        img_list = torch.full_like(x, 0)
        vid = torch.full_like(v, 0)
        num_collected = 0
        for j, s in enumerate(scenes.values()):
            if num_collected == self.max_len:
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

            if self.config["model"] == "sum" or self.config["model"] == "distil" or self.config["model"] == "vid" or self.config["model"] == "pre_modal" or self.config["model"] == "sum_residual":
                if self.state == "train":
                    start_slice = randint(0, len(clip) - 13)
                    clip_slice = clip[start_slice:start_slice + 12]
                else:
                    start_slice = 0
                    clip_slice = clip[0:12]
                for i in range(12):
                    vid[num_collected][i] = self.vid_trans(
                        self.pil_loader(clip_slice[i]))
            img_t = self.img_trans(self.pil_loader(clip[randint(0, len(clip) -1)]))
            img_list[num_collected] = img_t
            #img_list = []
            num_collected += 1
        # vid = vid.permute(0, 2, 1, 3, 4)
        if self.config["model"] == "sum" or self.config["model"] == "distil" or self.config["model"] == "pre_modal" or self.config["model"] == "sum_residual":
            return label, img_list, vid
        if self.config["model"] == "frame":
            return label, img_list
        if self.config["model"] == "vid":
            return label, vid
        
