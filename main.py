import confuse 
from dataloaders.MIT_dl import MIT_RAW_Dataset
from models.contrastivemodel import SpatioTemporalContrastiveModel, NT_Xent
from torch.utils.data import DataLoader
import torch
from transforms import img_transforms, spatio_cut, audio_transforms
from models import models


def train(config):
    bs = config["batch_size"].get()
    gpu = config["device"].get()
    device = torch.device(f"cuda:{gpu}")
    model = SpatioTemporalContrastiveModel(config)
    # dataset = CustomDataset(config)
    dataset = MIT_RAW_Dataset(config)
    print(len(dataset))
    train_loader = DataLoader(dataset, bs, shuffle=False, drop_last=True)
    for i, d in enumerate(train_loader):
        print(d["x_i"]["video"].shape)


def main():
    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("config.yaml")
    train(config)


if __name__ == "__main__":
    main()
