import pandas as pd
import csv
import torch
import os
from torch.utils.data import Dataset

LABEL = "label"
CHUNK = "chunk"
IMG_DIR = "imgs"
IMG_EMBED = "Eimage"
LOC_EMBED = "Elocation"
VID_EMBED = "Evideo"
DEPTH_EMBED = "Edepth"
AUDIO_EMBED = "Eaudio"


class CustomDataset(Dataset):
    def __init__(self, config):

        self.config = config
        self.data_frame = self.load_data()

    def load_data(self):
        data_frame = pd.read_csv(self.config['input_csv'].get(), chunksize=self.config['data_size'].get())
        print(data_frame)
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
