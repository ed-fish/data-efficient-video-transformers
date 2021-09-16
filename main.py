import confuse 
import wandb
from torchmetrics.functional import f1, auroc
from pytorch_lightning.plugins import DDPPlugin
from dataloaders.MIT_dl import MIT_RAW_Dataset, MITDataset, MITDataModule
from dataloaders.MMX_dl import MMX_Dataset, MMXDataModule
from models.contrastivemodel import SpatioTemporalContrastiveModel
from pytorch_lightning.loggers import WandbLogger
from models.basicmlp import BasicMLP
from torchmetrics.functional import f1, auroc
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.tensorboard import SummaryWriter
from callbacks.callbacks import SSLOnlineEval
from pytorch_lightning.callbacks import LearningRateMonitor
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint

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

#def test(config):

#    fine_tuner = SSLOnlineEval(z_dim=128, num_classes=15)
#    fine_tuner.to_device = to_device
#    model = SpatioTemporalContrastiveModel(config)
#    model = model.load_from_checkpoint(
#            config=config,
#            checkpoint_path="lightning_logs/version_64/checkpoints/test.ckpt",
#            hparams_file="lightning_logs/version_64/hparams.yaml",
#            map_location=None
#            )

#    log_callback = LogCallback()
#    lr_monitor = 
#    callbacks=[fine_tuner, log_callback, ]

#    dm = MMXDataModule("data_processing/mmx_tensors_train.pkl","data_processing/mmx_tensors_val.pkl", config)
#    trainer = pl.Trainer(gpus=[1], callbacks=callbacks)
#    #.test(model, datamodule=dm)

def main():

    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("config.yaml")

    wandb_logger = WandbLogger(project="MMX_Scene_Contrastive", log_model='all')
    bs = config["batch_size"].get()
    aggregation = config["aggregation"].get()

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    learning_rate = config["learning_rate"].get()
    train_experts = config["train_experts"].get()
    test_experts = config["test_experts"].get()
    momentum = config["momentum"].get()
    weight_decay = config["weight_decay"].get()
    warm_up = config["warm_up"].get()
    num_samples = config["num_samples"].get()
    device = config["device"].get()
    epochs = config["epochs"].get()


    params = { "batch_size": bs,
               "input_shape":config["input_shape"].get(),
               "hidden_layer":config["hidden_layer"].get(),
               "projection_size":config["projection_size"].get(),
               "output_shape": config["output_shape"].get(),
               "train_experts":train_experts,
               "test_experts": test_experts,
               "learning_rate": learning_rate,
               "aggregation": aggregation,
               "weight_decay": weight_decay,
               "momentum": momentum,
               "warm_up": warm_up,
               "num_samples": num_samples,
               "device": device,
               "epochs": epochs}

    wandb.init(project="contrastive-scene", name="mmx-video-full", config=params)
    config = wandb.config

    model = SpatioTemporalContrastiveModel(config)
    # model = BasicMLP(config)
    fine_tuner = SSLOnlineEval(z_dim=128, num_classes=15)
    fine_tuner.to_device = to_device

    # model = BasicMLP(config)
    # dataset = CustomDataset(config)
    # dm = MMXDataModule("data_processing/mmx_tensors_train.pkl","data_processing/mmx_tensors_val.pkl", config)
    dm = MMXDataModule("data_processing/scene_temporal/mmx_tensors_train_3.pkl","data_processing/scene_temporal/mmx_tensors_val_3.pkl", config)
    # checkpoint = ModelCheckpoint(monitor="train/contrastive/loss")

    # train_dataset = CSV_Dataset(config, test=False)
    # val_dataset = CSV_Dataset(config, test=True)
 
    # train loader - mmx_tensors_train.pkl
    # val loader - mmx_tensors_val.pkl

    # train_loader = DataLoader(train_dataset, bs, shuffle=True, collate_fn=custom_collater, num_workers=1, drop_last=True)
    # val_loader = DataLoader(val_dataset, bs, shuffle=False, collate_fn=custom_collater, num_workers=0, drop_last=True)

    # trainer = pl.Trainer(gpus=1, max_epochs=100,callbacks=[LogCallback()])
    checkpoints = ModelCheckpoint(monitor="train/contrastive/loss",
                                  dirpath="weights/contrastive-scene/",
                                  filename=''.join(config['train_experts']),
                                  mode="min"
                                  )

    callbacks = [fine_tuner, lr_monitor, checkpoints]
    trainer = pl.Trainer(gpus=[config["device"]], max_epochs=config["epochs"], callbacks=callbacks, logger=wandb_logger, accelerator="ddp")
    trainer.fit(model, datamodule=dm)
    #trainer.test(model, datamodule=dm, ckpt_path="lightning_logs/version_64/checkpoints/test.ckpt")
    # test(config)

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main()


