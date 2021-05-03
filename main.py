import confuse
from dataloaders.dataloader import CustomDataset
from models.contrastivemodel import SpatioTemporalContrastiveModel, NT_Xent
from torch.utils.data import DataLoader
import torch
# def pool_stack(tensor_stack):
#   pooled_output =


def train(config):
    bs = config["batch_size"].get()
    gpu = config["device"].get()
    device = torch.device(f"cuda:{gpu}")
    model = SpatioTemporalContrastiveModel(config)
    dataset = CustomDataset(config)
    print(len(dataset))
    train_loader = DataLoader(dataset, bs, shuffle=False, drop_last=True)
    for i, d in enumerate(train_loader):
        embedding, output = model(d["motion"].squeeze(1).to(device))
        print(i, output.shape)


def main():
    config = confuse.Configuration("Movie-Spatio-Temp")
    config.set_file("config.yaml")
    train(config)


if __name__ == "__main__":
    main()
