import tqdm
import pickle
import glob
import os
import re
import multiprocessing as mp
import numpy as np
import random


from collections import OrderedDict

# first create a list of all the filepaths to go through and collect - do only on one thread. 

def collect_labels(label):
    target_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                    'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Science Fiction', 'Thriller',  'War']
    label_list = np.zeros(15)

    for i, genre in enumerate(target_names):
        if genre == "Sci-Fi" or genre == "ScienceFiction":
            genre = "Science Fiction"
        if genre in label:
            label_list[i] = 1
    if np.sum(label_list) == 0:
        label_list[5] = 1

    return label_list
    
def label_tidy(label):
    if len(label) == 2:
        return collect_labels(label[0])
    else:
        return collect_labels(label)

def squish_folders(input_dir):
    all_files = []
    for genres in tqdm.tqdm(glob.glob(input_dir + "/*/")):
        for movies in os.listdir(genres):
            path = os.path.join(genres, movies)
            # use movies rather than individual scenes
            all_files.append(path)
    print("length of files", len(all_files))
    with open("cache.pkl", "wb") as cache:
        pickle.dump(all_files, cache)

def create_frame_path_dict(filepath):
    genre_name = filepath.split("/")[-3:-1]
    orig_dir = filepath.replace(
        "/mnt/bigelow/scratch/mmx_aug", "/mnt/fvpbignas/datasets/mmx_raw")

    if not os.path.exists(orig_dir):
        return False
   
    dirs = os.listdir(orig_dir)
    meta_data = os.path.join(orig_dir, dirs[0], "meta.pkl")

    with open(meta_data, "rb") as pickly:
        label = pickle.load(pickly)
    year = label[1]
    label = label[0]
    label = label_tidy(label)
    

    # subdirs = [000,001,002]
    scenes = glob.glob(filepath + "/*/")

    # this ensures all scenes are in order - some are "000" and "1" so regex checks this.
    scenes = sorted(scenes, key=lambda x: (
        int(re.findall("[0-9]+", x.split("/")[-2])[0])))
    if len(scenes) < 1:
        return False

    out_dict = OrderedDict()
    scene_dict = OrderedDict()

    for i, scene in enumerate(scenes):
        # chunks is a list of filepaths [001/001, 001/002]
        clips = glob.glob(scene + "/*/")
        if len(clips) < 1:
            continue

        clips = sorted(clips, key=lambda x: (
            int(re.findall("[0-9]+", x.split("/")[-2])[0])))
        clip_dict = OrderedDict()
        for j, clip in enumerate(clips):
            img_list = []
            img_dir = os.path.join(clip, "imgs")
            img_paths = glob.glob(img_dir + "/*")
            if len(img_paths) < 10:
                continue
            img_paths = sorted(img_paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
            while len(img_paths) < 16:
                img_paths.append(img_paths[-1])
            clip_dict[j] = img_paths
        scene_dict[i] = clip_dict
    out_dict = {"label": label,"year":year, "path":filepath, "scenes":scene_dict}
    return out_dict

def mp_handler():
    p = mp.Pool(40)
    data_list = []
    
    with open("cache.pkl", 'rb') as cache:
        data = pickle.load(cache)
        random.shuffle(data)
        # Remaining 80% to training set
        train_data = data[:int((len(data)+1)*.90)]
        # Splits 20% data to test set
        test_data = data[int((len(data)+1)*.90):]
        print("training_data", len(train_data))
        print("testing_data", len(test_data))

#squish_folders(input_dir)

    with open("mmx_train_temporal.pkl", 'ab') as pkly:
        for result in p.imap(create_frame_path_dict, tqdm.tqdm(train_data, total=len(train_data))):
            if result:
                pickle.dump(result, pkly)

    with open("mmx_val_temporal.pkl", 'ab') as pkly:
        for result in p.imap(create_frame_path_dict, tqdm.tqdm(test_data, total=len(test_data))):
            if result:
                pickle.dump(result, pkly)

if __name__ == "__main__":
    #rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    #resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    input_dir = "/mnt/bigelow/scratch/mmx_aug/"

    squish_folders(input_dir)
    mp_handler()

    

