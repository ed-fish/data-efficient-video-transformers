import pickle
import random

with open("mmx_tensors.pkl", "rb") as pklfile:
    data = []
    while 1:
        try:
            data.append(pickle.load(pklfile))
        except EOFError:
            break

random.shuffle(data)
train = data[:180000]
val = data[180000:]

print(len(train))
print(len(val))

print("dumping train")
with open("mmx_tensors_train.pkl", "ab") as train_fl:
    pickle.dump(train, train_fl)
print("dumping test")
with open("mmx_tensors_val.pkl", "ab") as val_fl:
    pickle.dump(val, val_fl)

