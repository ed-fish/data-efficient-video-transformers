import confuse
from dataloaders.MIT_dl import MIT_RAW_Dataset, CSV_Dataset
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
        
def custom_collater(batch):

    return {
            'label':[x['label'] for x in batch],
            'x_i_experts':[x['x_i_experts'] for x in batch],
            'x_j_experts':[x['x_j_experts'] for x in batch],
            'path':[x['path'] for x in batch]
            }


def train(config):
    bs = config["batch_size"].get()
    gpu = config["device"].get()
    device = torch.device(f"cuda:{gpu}")
    # model = SpatioTemporalContrastiveModel(config)
    model = BasicMLP(config)
    # dataset = CustomDataset(config)
    train_dataset = CSV_Dataset(config, test=False)
    val_dataset = CSV_Dataset(config, test=True)
    train_loader = DataLoader(train_dataset, bs, shuffle=True, collate_fn=custom_collater, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, bs, shuffle=False, collate_fn=custom_collater, num_workers=0, drop_last=True)

    # trainer = pl.Trainer(gpus=1, max_epochs=100,callbacks=[LogCallback()])
    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, train_loader,

def main():

    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("config.yaml")
    train(config)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
