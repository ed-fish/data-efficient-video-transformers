import glob
import pandas as pd
import random
import csv
import torch
import os
from collections import defaultdict
from torch.utils.data import Dataset
from transforms.img_transforms import ImgTransform
from transforms.spatio_cut import SpatioCut
from transforms.img_transforms import Normaliser
from models.pretrained.models import EmbeddingExtractor

LABEL = "label"
CHUNK = "chunk"
IMG_DIR = "imgs"
IMG_EMBED = "Eimage"
LOC_EMBED = "Elocation"
VID_EMBED = "Evideo"
DEPTH_EMBED = "Edepth"
AUDIO_EMBED = "Eaudio"


class MIT_RAW_Dataset(Dataset):
    def __init__(self, config, pre_computed=True):
        self.config = config
        self.pre_computed = pre_computed
        self.chunk_size = config['data_size'].get()
        self.data_frame = self.load_data()
        self.ee = EmbeddingExtractor(self.config)

    def load_data(self):
        train_data_frame = pd.read_csv(self.config['train_csv'].get())
        # val_data_frame = pd.read_csv(self.config['val.csv'].get())
        return train_data_frame

    def __len__(self):
        return len(self.data_frame)

    def stack_and_permute_vid(self, img_list):
        img_list = torch.stack(img_list)
        img_list = img_list.squeeze(1)
        img_list = img_list.permute(1, 0, 2, 3)
        return img_list

    def open_pt_return_list(self, folder_path):
        print(folder_path)
        items = glob.glob(folder_path + "/*.pt")
        tensor_list = []
        print(items)
        if len(items) > 1:
            for i in items:
                x = torch.load(i)
                tensor_list.append(x)
            return tensor_list
        else:
            print(items[0])
            return torch.load(items[0])

    # For precomputed embeddings that need to be loaded
    def collect_pre_computed_embeddings(self, video, config, label):
        sample_dict = defaultdict(dict)
        video_name = os.path.basename(video).replace(".mp4", "")
        root_dir = os.path.join(config["train_root"].get(), label, video_name)
        dirs = glob.glob(root_dir + "/*/")
        print(dirs)
        x_i_folder = dirs.pop(random.randrange(len(dirs)))
        x_j_folder = dirs.pop(random.randrange(len(dirs)))
        for s in ["x_i", "x_j"]:
            if s == "x_i":
                x_folder = x_i_folder
            else:
                x_folder = x_j_folder

            sample_dict[s]["video"] = self.open_pt_return_list(os.path.join(x_folder, "video-embeddings"))
            sample_dict[s]["location"] = self.open_pt_return_list(os.path.join(x_folder, "location-embeddings"))
            sample_dict[s]["image"] = self.open_pt_return_list(os.path.join(x_folder, "img-embeddings"))
        return sample_dict


    def collect_embedding(self, video, config):
        norm = Normaliser(config)
        sc = SpatioCut()
        video_imgs = sc.cut_vid(video, 16)

        # Take two groups of frames randomly - may want to make this
        # a temporal distance in the future as per the spatio-temporal
        # paper.

        x_i = video_imgs.pop(random.randrange(0, len(video_imgs)))
        x_j = video_imgs.pop(random.randrange(0, len(video_imgs)))
        augment_i = ImgTransform(x_i[0], config)
        augment_j = ImgTransform(x_j[0], config)
        sample_dict = defaultdict(dict)
        i_3d, i_loc, i_obj = [], [], [], []
        j_3d, j_loc, j_obj = [], [], [], []

        for img in x_i:
            t_img = augment_i.transform_with_prob(img)
            i_3d.append(norm.video_model(t_img))
            i_loc.append(norm.location_model(t_img))
            # i_dep.append(norm.depth_model(t_img))
            i_obj.append(norm.img_model(t_img))

        i_3d = self.stack_and_permute_vid(i_3d)
        sample_dict["x_i"]["video"] = i_3d
        sample_dict["x_i"]["location"] = i_loc
        # sample_dict["x_i"]["depth"] = i_dep
        sample_dict["x_i"]["image"] = i_obj

        for img in x_j:
            t_img = augment_j.transform_with_prob(img)
            j_3d.append(norm.video_model(t_img))
            j_loc.append(norm.location_model(t_img))
            # j_dep.append(norm.depth_model(t_img))
            j_obj.append(norm.img_model(t_img))

        j_3d = self.stack_and_permute_vid(j_3d)
        sample_dict["x_j"]["video"] = j_3d
        sample_dict["x_j"]["location"] = j_loc
        # sample_dict["x_j"]["depth"] = i_dep
        sample_dict["x_j"]["image"] = j_obj

        return sample_dict

    def __getitem__(self, idx):
        embed_dict = dict()
        embed_dict["label"] = self.data_frame.at[idx, "label"]
        embed_dict["video"] = self.data_frame.at[idx, "path"]

        if self.pre_computed:
            embedding_dict = self.collect_pre_computed_embeddings(embed_dict["video"], self.config, embed_dict["label"])
            x_i = embedding_dict["x_i"]
            x_j = embedding_dict["x_j"]

            for key, value in x_i.items():
                x_i[key] = self.ee.return_expert_for_key_pretrained(key, value)

            for key, value in x_j.items():
                x_j[key] = self.ee.return_expert_for_key_pretrained(key, value)
            return {'label': embedding_dict['label'], 'x_i': embedding_dict["x_i"], 'x_j': embedding_dict["x_j"]}

        else:
            embedding_dict = self.collect_embedding(embed_dict["video"], self.config)
            x_i = embedding_dict["x_i"]
            x_j = embedding_dict["x_j"]

            for key, value in x_i.items():
                x_i[key] = self.ee.return_expert_for_key(key, value)

            for key, value in x_j.items():
                x_j[key] = self.ee.return_expert_for_key(key, value)

            return { 'label': embed_dict['label'], 'x_i': x_i, 'x_j': x_j }


class CustomDataset(Dataset):
    def __init__(self, config):

        self.config = config
        self.data_frame = self.load_data()

    def load_data(self):
        data_frame = pd.read_csv(self.config['input_csv'].get(), chunksize=self.config['data_size'].get())
        return data_frame

    # def stack(self):

    def __len__(self):
        return len(self.data_frame)

    def collect_embeddings(self, data_type, idx):
        embedding_stack = []
        data_path = self.data_frame.at[idx, data_type]
        if len(os.listdir(data_path)) > 1:
            for embed in os.listdir(data_path):
                embed_path = os.path.join(data_path, embed)
                embedding_stack.append(torch.load(embed_path))
            data = torch.stack(embedding_stack)
        else:
            for embed in os.listdir(data_path):
                embed_path = os.path.join(data_path, embed)
            data = torch.load(embed_path)

        return data


    def __getitem__(self, idx):
        embed_dict = dict()
        embed_dict["label"] = self.data_frame.at[idx, LABEL]
        embed_dict["chunk"] = self.data_frame.at[idx, CHUNK]
        expert_list = self.config['experts'].get()
        if "image" in expert_list:
            embed_dict["image"] = self.collect_embeddings(IMG_EMBED, idx)
        if "location" in expert_list:
            embed_dict["location"] = self.collect_embeddings(LOC_EMBED, idx)
        if "depth" in expert_list:
            embed_dict["depth"] = self.collect_embeddings(DEPTH_EMBED, idx)
        if "motion" in expert_list:
            embed_dict["motion"] = self.collect_embeddings(VID_EMBED, idx)
        if "audio" in expert_list:
            embed_dict["audio"] = self.collect_embeddings(AUDIO_EMBED, idx)

        return embed_dict

