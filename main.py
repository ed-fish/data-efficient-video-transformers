import confuse
from dataloaders.MIT_dl import MIT_RAW_Dataset, CSV_Dataset
from models.contrastivemodel import SpatioTemporalContrastiveModel
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
    model = SpatioTemporalContrastiveModel(config)
    # dataset = CustomDataset(config)
    dataset = CSV_Dataset(config)
    train_loader = DataLoader(dataset, bs, shuffle=False, collate_fn=custom_collater, num_workers=0, pin_memory=True, drop_last=True)
    trainer = pl.Trainer(gpus=1, max_epochs=100,callbacks=[LogCallback()])
    # trainer.fit(model, train_loader)
    trainer.test(model, train_loader,
                 ckpt_path='/home/ed/self-supervised-video/lightning_logs/version_39/checkpoints/epoch=1-step=781.ckpt')

def main():

    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("config.yaml")
    train(config)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
