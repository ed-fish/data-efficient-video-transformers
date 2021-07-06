import confuse
from dataloaders.MIT_dl import MIT_RAW_Dataset, CSV_Dataset, MMXDataModule
from models.contrastivemodel import SpatioTemporalContrastiveModel
from models.basicmlp import BasicMLP
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.tensorboard import SummaryWriter
import torchvision
# from transforms import img_transforms, spatio_cut, audio_transforms

class LogCallback(Callback):
    def on_test_end(self, trainer, pl_module):
        print("ended test")
        writer = SummaryWriter()
        embeddings = torch.stack(pl_module.proj_list)
        imgs = pl_module.img_list
        print(len(imgs))
        imgs = torch.stack(imgs)
        print(imgs.shape)
        writer.add_embedding(embeddings, label_img=imgs)
        print("added projection")
        

def train(config):
    bs = config["batch_size"].get()
    gpu = config["device"].get()
    device = torch.device(f"cuda:{gpu}")
    # model = SpatioTemporalContrastiveModel(config)
    model = BasicMLP(config)
    # dataset = CustomDataset(config)
    dm = MMXDataModule("mmx_tensors_train.pkl", config)

    # train_dataset = CSV_Dataset(config, test=False)
    # val_dataset = CSV_Dataset(config, test=True)
   
    # train loader - mmx_tensors_train.pkl
    # val loader - mmx_tensors_val.pkl

    # train_loader = DataLoader(train_dataset, bs, shuffle=True, collate_fn=custom_collater, num_workers=1, drop_last=True)
    # val_loader = DataLoader(val_dataset, bs, shuffle=False, collate_fn=custom_collater, num_workers=0, drop_last=True)

    # trainer = pl.Trainer(gpus=1, max_epochs=100,callbacks=[LogCallback()])

    trainer = pl.Trainer(gpus=[1], max_epochs=100)
    trainer.fit(model, dm)
    # trainer.test(model, train_loader,

def main():

    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("config.yaml")
    train(config)

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()

