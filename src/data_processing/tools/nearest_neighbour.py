import pickle
import os
import re
import random
import numpy as np
from annoy import AnnoyIndex
import pandas as pd
import streamlit as st
from youtubesearchpython import VideosSearch

with open("embed_dict.pkl", "rb") as file:
    data_base = pickle.load(file)

st.title('Movie trailer recommendation')
st.text("Approximate nearest neighbours using only visual features (no metadata!)")
st.text("Embeddings extracted from a custom video transformer encoder.")

def annoy_processor(random_choice=False, id_n=0):
    # # 4096
    # from annoy import annoyindex
    recall = {}

    f = 15
    t = AnnoyIndex(f, 'euclidean')
    print(len(data_base.keys()))
    for i in range(len(data_base.keys())):
        v = data_base[i]["embedding"]
        t.add_item(i, v)

    t.build(750)
    t.save('./test.ann')
    u = AnnoyIndex(15, 'euclidean')
    u.load("./test.ann")
    if random_choice:
        results = u.get_nns_by_item(random.randrange(299), 10)
    else:
        results = u.get_nns_by_item(id_n, 10)
    for i, x in enumerate(results):
        recall[i] = {"path":data_base[x]["path"], "name": os.path.basename(
            os.path.normpath(data_base[x]["path"])), "actual": data_base[x]["actual"], "predicted": data_base[x]["predicted"]}
    #recall = pd.DataFrame.from_dict(recall, orient="index")
    return recall
 
def load_data(nrows):
    data = pd.DataFrame.from_dict(data_base, orient="index")
    return data

def retrieve_movies(random_choice=False, id_n=0):
    st.subheader("10 similar movies")
    data = annoy_processor(random_choice, id_n)
#    st.write(annoy_processor())
    col1, col2 = st.columns(2)
    cols = [col1, col2]

    for i in range(10):
        # name = re.sub(r"(\w)([A-Z])", r"\1 \2",data[i]["name"])
        cols[i%len(cols)].write(data[i]["path"])
        cols[i%len(cols)].image(data[i]["path"])
        # video_search = VideosSearch(name + " movie trailer", limit = 1)
        # result = video_search.result()
        # try:
        #     url = result["result"][0]['link']
        #     cols[i%len(cols)].video(url)
        # except IndexError:
        #     cols[i%len(cols)].write("no video")

        cols[i%len(cols)].caption("Actual genre:" + str(data[i]["actual"]))
        # cols[i%len(cols)].write(str(data[i]["actual"]))
        cols[i%len(cols)].caption("Predicted genre:" + str(data[i]["predicted"]))
            # cols[i%len(cols)].write(str(data[i]["predicted"]))

def tsne_projection():
    writer = SummaryWriter(log_dir="summaries")
    embedding = np.stack([data_base[i]["embedding"]
                          for i in range(len(data_base.keys()))])
    print(embedding.shape)
    keys = [data_base[i]["path"] for i in range(len(data_base.keys()))]
    print(len(keys))
    writer.add_embedding(embedding, metadata=keys)

data_load_state = st.text("loading data")
data = load_data(len(data_base.keys()))
data_load_state.text("loading data... done!")

# st.subheader('Raw data')
# st.write(data)

option = st.selectbox("pick a trailer from the drop down", data)
if st.button("generate random cluster"):
    retrieve_movies(random_choice=True, id_n=0)
if st.button("search with selected"):
    id_n = data.index[data["path"] == option].tolist()[0]
    print(id_n)
    retrieve_movies(random_choice=False, id_n=id_n)