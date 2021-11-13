import confuse
import pytorch_lightning as pl
from pytorch_lightning.utilities import distributed
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torchmetrics.functional import f1, auroc
from pytorch_lightning.loggers import WandbLogger
from dataloaders.MMX_Temporal_dl import MMXDataset, MMXDataModule
from dataloaders.MIT_Temporal_dl import MITDataModule, MITDataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.LSTM import LSTMRegressor
from models.transformer import TransformerModel, SimpleTransformer
from callbacks.callbacks import TransformerEval, DisplayResults, MITEval
from yaml.loader import SafeLoader
import torch.nn as nn


def train():
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')
    wandb_logger = WandbLogger(
        project="self-supervised-video")
    transformer_callback = TransformerEval()
    # checkpoint = ModelCheckpoint(
    #     save_top_k=-1, dirpath="trained_models/mmx/double", filename="double-{epoch:02d}")
    display = DisplayResults()

    # dm = MITDataModule("data/mit/mit_tensors_train_wc.pkl","data/mit/mit_tensors_train_wc.pkl", config)
    # dm = MMXDataModule("data/mmx/mmx_tensors_val.pkl","data/mmx/mmx_tensors_val.pkl", config)
    # configuration

# Open the file and load the file
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
    wandb.init(project="transformer-video", name="mit-all-sgd",
               config=data)
    config = wandb.config
    #callbacks = [checkpoint, transformer_callback, display]
    miteval = MITEval()
    callbacks = [miteval]

    model = SimpleTransformer(**config)

    # lstm
    # model = LSTMRegressor(seq_len=200, batch_size=64, criterion=nn.BCELoss(
    # ), n_features=4608, hidden_size=512, num_layers=4, dropout=0.2, learning_rate=0.00005)

    # MMX TRAINER
    # trainer = pl.Trainer(gpus=[3], callbacks=[
    #     transformer_callback], logger=wandb_logger)

    # MIT TRAINER
    trainer = pl.Trainer(gpus=[config["device"]], callbacks=callbacks, logger=wandb_logger)

    # MMX DATASET

    # dm = MMXDataModule("data/mmx/mmx_train_temporal.pkl",
    #                   "data/mmx/mmx_val_temporal.pkl", config)

    # MIT DATASET
    dm = MITDataModule("data/mit/MIT_train_temporal.pkl",
                       "data/mit/MIT_validation_temporal.pkl", config)
    trainer.fit(model, datamodule=dm)
    # model = model.load_from_checkpoint(
    #     "trained_models/mmx/double/double-epoch=127-v1.ckpt", **config)

    # trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    train()
