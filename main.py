import confuse
from dataloaders.MIT_dl import MIT_RAW_Dataset, MITDataset, MITDataModule
from models.contrastivemodel import SpatioTemporalContrastiveModel, OnlineEval
from models.basicmlp import BasicMLP
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.tensorboard import SummaryWriter
from callbacks.callbacks import SSLOnlineEval
import torchvision
# from transforms import img_transforms, spatio_cut, audio_transforms

class LogCallback(Callback):
    def on_test_end(self, trainer, pl_module):
        print("ended test")
        writer = SummaryWriter()
        embeddings = torch.stack(pl_module.proj_list)
        # imgs = pl_module.img_list
        labels = pl_module.label_list
        # print(len(imgs))
        # imgs = torch.stack(imgs)
        # print(imgs.shape)
        writer.add_embedding(embeddings, metadata=labels)
        print("added projection")
        
def to_device(batch, device):
    x_i_experts = batch["x_i_experts"]
    label = batch["label"]
    x_i_experts = [torch.cat(x, dim=-1) for x in x_i_experts]
    x_i_experts = torch.stack(x_i_experts)
    label = torch.stack(label)
    x_i_experts = x_i_experts.to(device)
    label = label.to(device)
    return x_i_experts, label


def train(config):
    bs = config["batch_size"].get()
    gpu = config["device"].get()
    device = torch.device(f"cuda:{gpu}")
    model = SpatioTemporalContrastiveModel(config)
    # model = BasicMLP(config)
    fine_tuner = SSLOnlineEval(z_dim=1024, num_classes=305)
    fine_tuner.to_device = to_device

    callbacks = [fine_tuner]
    # model = BasicMLP(config)
    # dataset = CustomDataset(config)
    # dm = MMXDataModule("data_processing/mmx_tensors_train.pkl","data_processing/mmx_tensors_val.pkl", config)
    dm = MITDataModule("data_processing/mit_tensors_train_wc.pkl","data_processing/mit_tensors_train_wc.pkl", config)


    # train_dataset = CSV_Dataset(config, test=False)
    # val_dataset = CSV_Dataset(config, test=True)
 
    # train loader - mmx_tensors_train.pkl
    # val loader - mmx_tensors_val.pkl

    # train_loader = DataLoader(train_dataset, bs, shuffle=True, collate_fn=custom_collater, num_workers=1, drop_last=True)
    # val_loader = DataLoader(val_dataset, bs, shuffle=False, collate_fn=custom_collater, num_workers=0, drop_last=True)

    # trainer = pl.Trainer(gpus=1, max_epochs=100,callbacks=[LogCallback()])

    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(model, dm)
    #trainer.test(model, datamodule=dm, ckpt_path="lightning_logs/version_64/checkpoints/test.ckpt")

def test(config):

    fine_tuner = SSLOnlineEval(z_dim=1024, num_classes=305)
    fine_tuner.to_device = to_device
    
    model = SpatioTemporalContrastiveModel(config)
    model = model.load_from_checkpoint(
            config=config,
            checkpoint_path="lightning_logs/version_64/checkpoints/test.ckpt",
            hparams_file="lightning_logs/version_64/hparams.yaml",
            map_location=None
            )

    callback = LogCallback()
    callbacks=[fine_tuner, callback]

    dm = MMXDataModule("data_processing/mmx_tensors_train.pkl","data_processing/mmx_tensors_val.pkl", config)
    trainer = pl.Trainer(gpus=[1], callbacks=callbacks)
    trainer.test(model, datamodule=dm)

    
def main():

    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("config.yaml")
    train(config)
    # test(config)

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()

