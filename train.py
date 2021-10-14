import confuse
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torchmetrics.functional import f1, auroc
from pytorch_lightning.loggers import WandbLogger
from dataloaders.MMX_Temporal_dl import MMXDataset, MMXDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.transformer import TransformerModel
from callbacks.callbacks import TransformerEval, DisplayResults
from yaml.loader import SafeLoader


def train():
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')
    wandb_logger = WandbLogger(
        project="self-supervised-video", log_model='all')
    transformer_callback = TransformerEval()
    checkpoint = ModelCheckpoint(
        save_top_k=-1, dirpath="trained_models/mmx/double", filename="double-{epoch:02d}")
    display = DisplayResults()

    # dm = MITDataModule("data/mit/mit_tensors_train_wc.pkl","data/mit/mit_tensors_train_wc.pkl", config)
    # dm = MMXDataModule("data/mmx/mmx_tensors_val.pkl","data/mmx/mmx_tensors_val.pkl", config)
    # configuration

# Open the file and load the file
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
    wandb.init(project="transformer-video", name="testing",
               config=data)
    config = wandb.config

    model = TransformerModel(**config)
    trainer = pl.Trainer(gpus=[config["device"]], callbacks=[
                         checkpoint, transformer_callback, display], max_epochs=config["epochs"], logger=wandb_logger)

    dm = MMXDataModule("data/mmx/mmx_train.pkl",
                       "data/mmx/mmx_val.pkl", config)
    # trainer.fit(model, datamodule=dm)
    model = model.load_from_checkpoint(
        "trained_models/mmx/double/double-epoch=127-v1.ckpt", **config)

    trainer.test(model, datamodule=dm)


train()
