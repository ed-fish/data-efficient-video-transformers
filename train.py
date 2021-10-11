import confuse
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb
from torchmetrics.functional import f1, auroc
from pytorch_lightning.loggers import WandbLogger
from dataloaders.MIT_Temporal_dl import MITDataset, MITDataModule
from models.contrastivemodel import SpatioTemporalContrastiveModel
from dataloaders.MMX_Temporal_dl import MMXDataset, MMXDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.transformer import TransformerModel
from callbacks.callbacks import TransformerEval


# Dataloading
# The model expects shape of Sequence, Batch, Embedding
# For MIT the batches should be [3, 32, 1028] the total size of the vocabulary
def get_params():
    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("configs/transformer/mmx/default.yaml")
    bptt = config["batch_size"].get()
    save_path = config["save_path"].get()
    learning_rate = config["learning_rate"].get()
    scheduling = config["scheduling"].get()
    momentum = config["momentum"].get()
    weight_decay = config["weight_decay"].get()
    token_embedding = config["token_embedding"].get()
    experts = config["experts"].get()
    epochs = config["epochs"].get()
    n_warm_up = 70
    seq_len = config["seq_len"].get()
    ntokens = config["n_labels"].get()
    emsize = config["input_shape"].get()
    mixing_method = config["mixing_method"].get()
    nhid = 1850
    nlayers = 4
    frame_agg = config["frame_agg"].get()
    nhead = config["n_heads"].get()
    dropout = config["dropout"].get()
    frame_id = config["frame_id"].get()
    cat_norm = config["cat_norm"].get()
    cat_softmax = config["cat_softmax"].get()
    architecture = config["architecture"].get()
    device = config["device"].get()
    aggregation = config["aggregation"].get()
    cat_norm = config["cat_norm"].get()
    pooling = config["pooling"].get()

    params = {"experts": experts,
              "pooling": pooling,
              "output_shape": config["output_shape"].get(),
              "hidden_layer": config["hidden_layer"].get(),
              "device": config["device"].get(),
              "emsize": emsize,
              "cat_norm": cat_norm,
              "aggregation": aggregation,
              "input_shape": config["input_shape"].get(),
              "ntokens": ntokens,
              "mixing_method": mixing_method,
              "epochs": epochs,
              "frame_id": frame_id,
              "batch_size": bptt,
              "seq_len": seq_len,
              "nlayers": nlayers,
              "dropout": dropout,
              "cat_norm": cat_norm,
              "cat_softmax": cat_softmax,
              "nhid": nhid,
              "nhead": nhead,
              "n_warm_up": n_warm_up,
              "learning_rate": learning_rate,
              "scheduling": scheduling,
              "weight_decay": weight_decay,
              "momentum": momentum,
              "save_path": save_path,
              "token_embedding": token_embedding,
              "architecture": architecture,
              "frame_agg": frame_agg}
    return params
##### Training ####


def train():

    #    torch.multiprocessing.set_sharing_strategy('file_system')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    wandb_logger = WandbLogger(
        project="self-supervised-video", log_model='all')
    transformer_callback = TransformerEval()
    checkpoint = ModelCheckpoint(
        save_top_k=-1, dirpath="trained_models/mmx/double", filename="double-{epoch:02d}")

    # dm = MITDataModule("data/mit/mit_tensors_train_wc.pkl","data/mit/mit_tensors_train_wc.pkl", config)
    # dm = MMXDataModule("data/mmx/mmx_tensors_val.pkl","data/mmx/mmx_tensors_val.pkl", config)
    # configuration
    params = get_params()
    wandb.init(project="transformer-video",
               name="mmx-collab-w-pos", config=params)
    config = wandb.config
    dm = MMXDataModule("data/mmx/mmx_train.pkl",
                       "data/mmx/mmx_val.pkl", config)

    model = TransformerModel(config, config["ntokens"], config["emsize"], config["nhead"],
                             nhid=config["nhid"],
                             batch_size=config["batch_size"],
                             nlayers=config["nlayers"],
                             learning_rate=config["learning_rate"],
                             dropout=config["dropout"],
                             warmup_epochs=config["n_warm_up"],
                             max_epochs=config["epochs"],
                             seq_len=config["seq_len"],
                             token_embedding=config["token_embedding"],
                             architecture=config["architecture"],
                             mixing=config["mixing_method"])
    trainer = pl.Trainer(gpus=[config["device"]], callbacks=[
                         checkpoint, transformer_callback], max_epochs=config["epochs"], logger=wandb_logger)
    trainer.fit(model, datamodule=dm)


train()
