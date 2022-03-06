from typing import Iterable
import pandas as pd
from simplejson import OrderedDict
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
from random import randint
from collections import OrderedDict
import glob
from sklearn.utils import shuffle
import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class InputIterator(object):
    def __init__(self, data, batch_size_total, batch_size, seq_len, frame_len):
        super().__init__()
        self.data = data
        self.batch_size_total = batch_size_total
        self.batch_size = batch_size
        self.indices = list(range(len(self.data)))
        self.seq_len = seq_len
        self.frame_len = frame_len

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        batch = []
        targets = []
        for _ in range(self.batch_size):
            img_root = self.data["img_root"].iloc[self.i]
            labels = []
            vids = []
            for i in range(1, 7):
                labels.append(self.data[f"g{i}"].iloc[self.i])
            target = self.collect_labels(labels)
            scenes = sorted(glob.glob(img_root + "/*"))
            scene_len = len(scenes)

            for scene in range(self.seq_len):
                imgs = sorted(glob.glob(scenes[scene % scene_len] + "/*"))
                img_len = len(imgs)
                print("img len", img_len)
                for img in range(self.frame_len):
                    print("img img_len", img % img_len)
                    f = open(imgs[img % img_len], 'rb')
                    vid = np.frombuffer(f.read(), dtype=np.uint8)
                    batch.append(vid)
                    targets.append(target)
            self.i = (self.i + 1) % self.n
        print("BATHC", len(batch))
        return (batch, targets)

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


class SimplePipeline(Pipeline):
    def __init__(self, batch_size, eii,  num_threads=2, device_id=0, resolution=256, crop=224, is_train=True):
        super(SimplePipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12)
        self.source = ops.ExternalSource(source=eii, num_outputs=2)

    def define_graph(self):
        images, labels = self.source()
        images = fn.decoders.image(
            images, device="mixed", output_type=types.RGB)
        images = fn.resize(
            images,
            resize_shorter=fn.random.uniform(range=(120, 200)),
            interp_type=types.INTERP_LINEAR)
        images = fn.crop_mirror_normalize(
            images,
            crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
            crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
            dtype=types.FLOAT,
            crop=(112, 112),
            mean=[128., 128., 128.],
            std=[1., 1., 1.])
        return images, labels


class DALIClassificationLoader(DALIClassificationIterator):
    def __init__(
            self, pipelines, size=-1, reader_name=None, auto_reset=False, fill_last_batch=False, dynamic_shape=False, last_batch_padded=False):
        super().__init__(pipelines,
                         size,
                         reader_name,
                         auto_reset,
                         fill_last_batch,
                         dynamic_shape,
                         last_batch_padded)

        def __len__(self):
            batch_count = self._size // (self._num_gpus * self.batch_size)
            last_batch = 1 if self._fill_last_batch else 0
            print("COUNTER", batch_count)
            return batch_count + last_batch


class MMXLightDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, config):
        super().__init__()
        self.csv_path = csv_path
        self.config = config
        self.bs = config["batch_size"]
        self.seq_len = config["seq_len"]
        self.frame_len = config["frame_len"]
        self.bs_total = self.bs * self.seq_len * self.frame_len

    def setup(self, stage=None):

        self.data_frame = pd.read_csv(self.csv_path)
        self.data_frame = shuffle(self.data_frame)
        self.train_data = self.data_frame.iloc[:6047, :]
        self.train_data.reset_index(drop=True)
        self.val_data = self.data_frame.iloc[6047:6700, :]
        self.val_data.reset_index(drop=True)
        print(len(self.train_data))
        print(len(self.val_data))
        # device_id = self.local_rank
        # shard_id = self.global_rank
        # train_dataset = InputIterator(
        #     self.train_data, self.bs_total, self.bs, self.seq_len, self.frame_len)
        # val_dataset = InputIterator(
        #     self.val_data, self.bs_total, self.bs, self.seq_len, self.frame_len)

        # pipe_train = SimplePipeline(
        #     batch_size=self.bs_total, eii=train_dataset, num_threads=2, device_id=0)
        # pipe_train.build()
        # self.train_loader = DALIClassificationLoader(
        #     pipe_train, len(self.train_data), auto_reset=True)

        # pipe_val = SimplePipeline(
        #     batch_size=self.bs_total, eii=val_dataset, num_threads=2, device_id=0)
        # pipe_val.build()
        # self.val_loader = DALIClassificationLoader(
        #     pipe_train, len(self.val_data), auto_reset=True)
        self.train_loader = DataLoader(MMXLightDataset(self.train_data, self.config, state="train"), batch_size=self.bs, drop_last=True, num_workers=10, pin_memory=True)
        self.val_loader = DataLoader(MMXLightDataset(self.val_data, self.config, state="val"), batch_size=self.bs, drop_last=True, num_workers=10, pin_memory=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.val_loader


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
            #transforms.RandomErasing(),
        ])

        self.val_vid = transforms.Compose([
            transforms.Resize(112),
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
        # img_tensor = img_tensor.float()
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
                k = k % img_len
            #img_list[i] = self.img_trans(self.pil_loader(imgs[3]))
            i += 1
            i = i % scene_len

        return target, img_list, vid

        # Add looping for out of index samples
        # grab img as well as video frames or find model with same dim
        # set scenes etc as a parameter - not hardcoded
