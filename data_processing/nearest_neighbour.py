import pickle
import os
import random
import torch
import numpy as np
from annoy import AnnoyIndex
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import streamlit as st


with open("/home/ed/self-supervised-video/data_processing/embed_dict.pkl", "rb") as file:
    data_base = pickle.load(file)

st.title('Movie trailer reccomendation')


def annoy_processor():
    # # 4096
    # from annoy import AnnoyIndex

    recall = {}

    f = 4096
    t = AnnoyIndex(f, 'euclidean')
    print(len(data_base.keys()))
    for i in range(len(data_base.keys())):
        v = data_base[i]["embedding"]
        t.add_item(i, v)

    t.build(100)

    t.save('/home/ed/self-supervised-video/data_processing/test2.ann')
    u = AnnoyIndex(4096, 'euclidean')
    u.load("/home/ed/self-supervised-video/data_processing/test2.ann")
    results = u.get_nns_by_item(random.randrange(500), 10)
    for i in results:
        recall[i] = {"name": os.path.basename(
            os.path.normpath(data_base[i]["path"])), "actual": data_base[i]["actual"], "predicted": data_base[i]["predicted"]}

        print(data_base[i]["path"])
        print(data_base[i]["actual"])
        print(data_base[i]["predicted"])
    recall = pd.DataFrame.from_dict(recall, orient="index")
    return recall


@st.cache
def load_data(nrows):
    data = pd.DataFrame.from_dict(data_base, orient="index")
    return data


data_load_state = st.text("loading data")
data = load_data(len(data_base.keys()))
data_load_state.text("loading data... done!")

st.subheader('Raw data')
st.write(data)

if st.checkbox("select random comparrison"):
    st.subheader("random choice")
    st.write(annoy_processor())


def tsne_projection():
    writer = SummaryWriter(log_dir="summaries")
    embedding = np.stack([data_base[i]["embedding"]
                          for i in range(len(data_base.keys()))])
    print(embedding.shape)
    keys = [data_base[i]["path"] for i in range(len(data_base.keys()))]
    print(len(keys))
    writer.add_embedding(embedding, metadata=keys)
