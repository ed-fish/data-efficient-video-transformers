import confuse
from dataloaders.MIT_dl import MIT_RAW_Dataset, CSV_Dataset
from models.contrastivemodel import SpatioTemporalContrastiveModel
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
# from transforms import img_transforms, spatio_cut, audio_transforms


def train(config):
    bs = config["batch_size"].get()
    gpu = config["device"].get()
    device = torch.device(f"cuda:{gpu}")
    model = SpatioTemporalContrastiveModel(config)
    # dataset = CustomDataset(config)
    dataset = CSV_Dataset(config)
    train_loader = DataLoader(dataset, bs, shuffle=False, num_workers=0, pin_memory=True)
    # for label, data_x, data_y in train_loader:
    #     model.training_step(label, data_x, data_y)

    trainer = pl.Trainer(gpus=4, max_epochs=100, accelerator="ddp")
    trainer.fit(model, train_loader)

def main():

    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("config.yaml")
    train(config)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
