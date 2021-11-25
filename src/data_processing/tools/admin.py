import pickle as pkl
import re
import fnmatch

def pickleLoader(pklFile):
    try:
        while True:
            yield pkl.load(pklFile)
    except EOFError:
        pass

with open("mmx_tensors_train.pkl", "rb") as pkly:
    for entry in pickleLoader(pkly):
        if "Horror/TheWolfman" in entry["path"]:
            print(entry["path"])
            pass
        else:
            with open("mmx_tensors_train_3.pkl", 'ab') as pkly:
                pkl.dump(entry, pkly)

 
        
