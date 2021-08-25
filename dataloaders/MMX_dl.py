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
        self.bs = self.config["batch_size"].get()

    def custom_collater(self, batch):

        return {
                'label':[x['label'] for x in batch],
                'x_i_experts':[x['x_i_experts'] for x in batch],
                'x_j_experts':[x['x_j_experts'] for x in batch],
                'path':[x['path'] for x in batch]
                }

    # def prepare_data(self):
    #    data = self.load_data(self.pickle_file)
    #    self.data = self.clean_data(data)

    def clean_data(self, data_frame):

        print("cleaning data")
        print(len(data_frame))
        id_to_drop = []
        for i in range(len(data_frame)):
            
            data = data_frame.at[i, "data"]
            data_chunk = list(data.values())           
            if len(data_chunk) == 0:
                print("dropping index with no data", i, len(data_chunk))
                data_frame = data_frame.drop(i)
                continue

            x = [len(data) for data in data_chunk]
            if sum(x) < len(x) * 3:
                print("dropping index with incomplete data", i, len(data))
                data_frame = data_frame.drop(i)
                continue

                # test = []
                # for f in data[1]: # data1 == img_embeddings, data2 == motion?, data0=location
                 #   print(f)
                 #   f = torch.load(f)
                  #  f = f.squeeze()
                   # test.append(f)
                    # print(f.dim)
                    # if f.dim() > 0:
                    #    test.append(f)
                    #else:
                    #    data_frame = data_frame.drop(i)
                    #    continue
                # try:
                #    test = torch.cat(test, dim=-1)
                # except:
                #    data_frame = data_frame.drop(i)
                #    print("dropping", i)
                #    continue
                #print(test.shape[0])
                #if test.shape[0] != 2560:
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
                    #data = pickle.load(pkly)
                except EOFError:
                    break

        data_frame = pd.DataFrame(data)
        print("data loaded")
        print("length", len(data_frame))
        # data_frame = data_frame.head(10000)
        return data_frame

    def setup(self, stage):

        self.train_data = self.load_data(self.train_data)
        self.train_data = self.clean_data(self.train_data)

        self.val_data = self.load_data(self.val_data)
        self.val_data = self.clean_data(self.val_data)

    def train_dataloader(self):
        return DataLoader(MMX_Dataset(self.train_data, self.config, train=True), self.bs, shuffle=True, collate_fn=self.custom_collater, num_workers=0, drop_last=True)

    def val_dataloader(self):
        return DataLoader(MMX_Dataset(self.val_data, self.config, train=False), self.bs, shuffle=False, collate_fn=self.custom_collater, num_workers=0, drop_last=True)
    # For now use validation until proper test split obtained
    def test_dataloader(self):
        return DataLoader(MMX_Dataset(self.train_data, self.config), 1, shuffle=False, collate_fn=self.custom_collater, num_workers=0)



class MMX_Dataset(Dataset):
    def __init__(self, data, config, train=True):
        super().__init__()

        self.config = config
        self.data_frame = data
        self.aggregation = self.config["aggregation"].get()
        self.train = train


    def __len__(self):
        return len(self.data_frame)
    
    def collect_labels(self, label):
	
        target_names = ['Action',
                        'Adventure',
                        'Animation',
                        'Biography',
                        'Comedy',
                        'Crime',
                        'Documentary',
                        'Drama',
                        'Family',
                        'Fantasy',
                        'History',
                        'Horror',
                        'Music',
                        'Mystery',
                        'Romance',
                        'Science Fiction',
                        'Short',
                        'Sport',
                        'Thriller',
                        'TV Movie',
                        'War',
                        'Western',
                        ]

        label_list = np.zeros(22)

        for i, genre in enumerate(target_names):
            if genre == "Sci-Fi" or genre == "ScienceFiction":
                genre = "Science Fiction"
            if genre in label:
                label_list[i] = 1

        return label_list


    def __getitem__(self, idx):

        label =  self.data_frame.at[idx, "label"]
        label = self.collect_labels(label[0])
        label = torch.FloatTensor(label)
        data = self.data_frame.at[idx, "data"]
        path = self.data_frame.at[idx, "path"]
        path = path.replace("/mnt/fvpbignas/datasets/mmx_raw", "/mnt/bigelow/scratch/mmx_aug")
        try:
            path = glob.glob(path + "/*/")[0]
            path = os.path.join(path, "imgs")
            path = glob.glob(path + "/*")[1]
        except:
            path = "None"
        scene = self.data_frame.at[idx, "scene"]

        experts_xi = []
        experts_xj = []

        # apply mix-up if less than 2 samples
        # keys here are the scenes ["000", "001", "002"] etc
        # if there are less than 2 scenes in the sample we need to do something else. 

        if self.train:
            experts = self.config["train_experts"].get()
        else:
            experts = self.config["test_experts"].get()
        if len(data) < 2:
            data = list(data.values())[0] # take the first index as its the only valid one
            #data = list(data.values())
            if idx == 0:
                idmx = idx + 1
            else:
                idmx = idx - 1
            mix_up_data = self.data_frame.at[idmx , "data"]
            mix_up_data = list(mix_up_data.values())[0] # take the first index of the sample either before or after e.g ["000"]

            # Now we have data = ["000"][experts] and mix_up_data = ["000"][experts]

            for expert in experts:
                expert_vec = data[expert]
                if not isinstance(expert_vec, str):
                    expert_vec = random.choice(expert_vec)
                expert_vec = torch.load(expert_vec)
                if len(expert_vec.shape) < 2:
                    expert_vec = expert_vec.unsqueeze(0)
                experts_xi.append(expert_vec)

                expert_vec = mix_up_data[expert]
                if not isinstance(expert_vec, str):
                    expert_vec = random.choice(expert_vec)
                expert_vec = torch.load(expert_vec)
                if len(expert_vec.shape) < 2:
                    expert_vec = expert_vec.unsqueeze(0)
                experts_xj.append(expert_vec)

        else:
            # select two random scenes as a positive pair
            x_i, x_j = random.sample(list(data.values()), 2)

            # x_i["test-location"] etc are valid keys

            for expert in experts:
                expert_vec = x_i[expert]
                if not isinstance(expert_vec, str):
                    expert_vec = random.choice(expert_vec)
                expert_vec = torch.load(expert_vec)
                if len(expert_vec.shape) < 2:
                    expert_vec = expert_vec.unsqueeze(0)
                experts_xi.append(expert_vec)

                expert_vec = x_j[expert]
                if not isinstance(expert_vec, str):
                    expert_vec = random.choice(expert_vec)
                expert_vec = torch.load(expert_vec)
                if len(expert_vec.shape) < 2:
                    expert_vec = expert_vec.unsqueeze(0)
                experts_xj.append(expert_vec)


        if self.aggregation == "debugging":
            experts_xi = torch.cat(experts_xi, dim=-1)
            experts_xj = torch.cat(experts_xj, dim=-1)

        return {"label":label, "path":path, "scene":scene, "x_i_experts":experts_xi, "x_j_experts":experts_xj}


class MIT_RAW_Dataset(Dataset):
    def __init__(self, config, pre_computed=True):
        super().__init__()
        self.config = config
        self.pre_computed = pre_computed
        self.chunk_size = config['data_size'].get()
        self.data_frame = self.load_data()
        # self.ee = EmbeddingExtractor(self.config)

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
        items = glob.glob(folder_path + "/*.pt")
        tensor_list = []
        if len(items) > 1:
            for i in items:
                with torch.no_grad():
                    x = torch.load(i, map_location="cuda:3")
                    x = x.detach()
                    tensor_list.append(x)
            return tensor_list
        else:
            with torch.no_grad():
                x = torch.load(items[0], map_location="cuda:3")
                x = x.detach()
                return x

    # For precomputed embeddings that need to be loaded
    def collect_pre_computed_embeddings(self, video, config, label):
        sample_dict = defaultdict(dict)
        # video_name = os.path.basename(video).replace(".mp4", "")
        # root_dir = os.path.join(config["train_root"].get())
        dirs = glob.glob(video + "/*/")
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

    """ def collect_embedding(self, video, config):
        norm = Normaliser(config)
        sc = SpatioCut()
        video_imgs = sc.cut_vid(video, 16)
        if len(video_imgs) < 16:
            return 0

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

        return sample_dict """


    def return_expert_for_key_pretrained(self, key, raw_tensor):

        if key == "image":
            if len(raw_tensor) > 1:
                output = torch.stack(raw_tensor)
                output = output.transpose(0, 2)
                output = F.adaptive_avg_pool1d(output, 1)
                output = output.transpose(1, 0).squeeze(2)
                output = output.squeeze(1)
            else:
                output = raw_tensor[0].unsqueeze(0)

        if key == "motion" or key == "video":
            output = raw_tensor[0].unsqueeze(0)

        if key == "location":
            if len(raw_tensor) > 1:
                output = torch.stack(raw_tensor)
                output = output.transpose(0, 2)
                output = F.adaptive_avg_pool1d(output, 1)
                output = output.transpose(1, 0).squeeze(2)
                output = output.squeeze(1)
            else:
                output = raw_tensor[0].unsqueeze(0)

        return output

    def __getitem__(self, idx):
        label =  self.data_frame.at[idx, "label"]
        path = self.data_frame.at[idx, "path"]
        
        if self.pre_computed:
            embedding_dict = self.collect_pre_computed_embeddings(path, self.config, label)

            x_i = embedding_dict["x_i"]
            x_j = embedding_dict["x_j"]

            for key, value in x_i.items():
                x_i[key] = self.return_expert_for_key_pretrained(key, value)

            for key, value in x_j.items():
                x_j[key] = self.return_expert_for_key_pretrained(key, value)

            return {'label': embedding_dict['label'], 'x_i': x_i, 'x_j': x_j}

        # else:
        #     embedding_dict = self.collect_embedding(embed_dict["video"], self.config)
        #     x_i = embedding_dict["x_i"]
        #     x_j = embedding_dict["x_j"]

        #     for key, value in x_i.items():
        #         x_i[key] = self.ee.return_expert_for_key(key, value)

        #     for key, value in x_j.items():
        #         x_j[key] = self.ee.return_expert_for_key(key, value)

        #     return { 'label': embed_dict['label'], 'x_i': x_i, 'x_j': x_j }


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

