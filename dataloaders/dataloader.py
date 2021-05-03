import pandas as pd
import csv
import torch
import os
from torch.utils.data import Dataset

NAME = "name"
GENRE = "genre"
SCENE = "scene"
YEAR = "year"
CHUNK = "chunk"
CLIP = "clip"
IMG_DIR = "imgs"
IMG_EMBED = "Eimage"
LOC_EMBED = "Elocation"
VID_EMBED = "Evideo"
DEPTH_EMBED = "Edepth"


class CustomDataset(Dataset):
    def __init__(self, config):

        self.config = config
        self.data_frame = self.load_data()

    def load_data(self):
        data_frame = pd.read_csv(self.config['input_csv'].get())
        # with open(self.config['input_csv'].get(), 'r') as csvfile:
        #csvreader = csv.reader(csvfile)
        #csv_fields = next(csvreader)
        #data_list = pd.DataFrame(csvfile, fields=csv_fields)
        print(data_frame)
        return data_frame

    # def stack(self):

    def __len__(self):
        return len(self.data_frame)

    def collect_embeddings(self, data_type, idx):
        embedding_stack = []
        data_path = self.data_frame.at[idx, data_type]
        for embed in os.listdir(data_path):
            embed_path = os.path.join(data_path, embed)
            embedding_stack.append(torch.load(embed_path))
        embedding_stack = torch.stack(embedding_stack)

        return embedding_stack

    # todo idx needs to be the scene location and then we retrieve the number of chunks and then we can use a temporal sampling interval method.

    def __getitem__(self, idx):
        embed_dict = dict()
        embed_dict["name"] = self.data_frame.at[idx, NAME]
        embed_dict["scene"] = self.data_frame.at[idx, SCENE]
        embed_dict["genre"] = self.data_frame.at[idx, GENRE]
        embed_dict["clip"] = self.data_frame.at[idx, CLIP]
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

        return embed_dict
