import pickle as pkl
import re
import fnmatch

def pickleLoader(pklFile):
    try:
        while True:
            yield pkl.load(pklFile)
    except EOFError:
        pass

with open("mit_tensors_train.pkl", "rb") as pkly:
    for entry in pickleLoader(pkly):
        if "fvpbignas" in entry["path"]:
            print(entry["path"])
            pass
        else:
            with open("mit_tensors_clean.pkl", 'ab') as pkly:
                pkl.dump(entry, pkly)

 
        
