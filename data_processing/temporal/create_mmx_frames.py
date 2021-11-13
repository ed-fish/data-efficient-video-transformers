import tqdm
import pickle
import glob
import os
import re

from collections import OrderedDict

# first create a list of all the filepaths to go through and collect - do only on one thread. 

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
   
    dirs = os.listdir(orig_dir)
    meta_data = os.path.join(orig_dir, dirs[0], "meta.pkl")
    with open(meta_data, "rb") as pickly:
        label = pickle.load(pickly)

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
            print(len(img_paths))
            clip_dict[j] = img_paths
        scene_dict[i] = clip_dict
    out_dict = {"label": label[0],"year":label[1], "path":filepath, "scenes":scene_dict}
    print(out_dict)
    return out_dict

input_dir = "/mnt/bigelow/scratch/mmx_aug/"
#squish_folders(input_dir)
with open("cache.pkl", 'rb') as cache:
    data = pickle.load(cache)
    for i, d in enumerate(data):
        create_frame_path_dict(d)

