import confuse 
import math
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb
import torchmetrics
from torchmetrics.functional import f1, auroc
from pytorch_lightning.loggers import WandbLogger
from dataloaders.MIT_Temporal_dl import MITDataset, MITDataModule
from models.contrastivemodel import SpatioTemporalContrastiveModel
from dataloaders.MMX_Temporal_dl import MMXDataset, MMXDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.transformer import TransformerModel
from callbacks.callbacks import TransformerEval
#from callbacks.callbacks import TransformerEval


# Dataloading
# The model expects shape of Sequence, Batch, Embedding
# For MIT the batches should be [3, 32, 1028] the total size of the vocabulary 
def get_params():

    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("config.yaml")
    bptt = config["batch_size"].get()
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
    nlayers = 3
    frame_agg = config["frame_agg"].get()
    nhead = config["n_heads"].get()
    dropout = config["dropout"].get()
    frame_id = config["frame_id"].get()
    cat_norm = config["cat_norm"].get()
    cat_softmax = config["cat_softmax"].get()
    architecture = config["architecture"].get()
    device = config["device"].get()
    aggregation=config["aggregation"].get()
    cat_norm = config["cat_norm"].get()

    params = { "experts":experts,
               "device":config["device"].get(),
               "emsize": emsize,
               "cat_norm":cat_norm,
               "aggregation":aggregation,
               "input_shape": config["input_shape"].get(),
               "ntokens": ntokens,
               "mixing_method":mixing_method,
               "epochs": epochs,
               "frame_id":frame_id,
               "batch_size": bptt,
               "seq_len": seq_len,
               "nlayers":nlayers,
               "dropout":dropout,
               "cat_norm": cat_norm,
               "cat_softmax": cat_softmax,
               "nhid":nhid,
               "nhead":nhead,
               "n_warm_up":n_warm_up,
               "learning_rate":learning_rate,
               "scheduling":scheduling,
               "weight_decay":weight_decay, 
               "momentum":momentum,
               "token_embedding":token_embedding,
               "architecture":architecture,
               "frame_agg":frame_agg }
    return params
##### Training ####

def train():
    os.system("taskset -p 0xff %d" % os.getpid())


    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    tr_eval = TransformerEval()

    wandb_logger = WandbLogger(project="self-supervised-video", log_model='all')
    # transformer_callback = TransformerEval()

    # dm = MMXDataModule("data/mmx/mmx_tensors_val.pkl","data/mmx/mmx_tensors_val.pkl", config)
    # configuration
    params = get_params()
    wandb.init(project="transformer-video", name="MIT-location", config=params)
    config = wandb.config

    dm = MITDataModule("data_processing/scene_temporal/mit_tensors_clean.pkl","data_processing/scene_temporal/mit_tensors_val.pkl", config)
    #dm = MMXDataModule("data_processing/trailer_temporal/mmx_tensors_train_3.pkl", "data_processing/trailer_temporal/mmx_tensors_val_3.pkl", config)
    #dm = MMXDataModule("data_processing/trailer_temporal/mmx_tensors_train_3.pkl", "data_processing/trailer_temporal/mmx_tens0ors_val_3.pkl", config)
    
    model = TransformerModel(config["ntokens"], config["emsize"], config["nhead"],
                             nhid = config["nhid"],
                             batch_size = config["batch_size"],
                             nlayers = config["nlayers"],
                             learning_rate = config["learning_rate"],
                             dropout = config["dropout"],
                             warmup_epochs = config["n_warm_up"], 
                             max_epochs = config["epochs"],
                             seq_len = config["seq_len"],
                             token_embedding = config["token_embedding"],
                             architecture = config["architecture"],
                             mixing = config["mixing_method"])
    trainer = pl.Trainer(gpus=[config["device"]], callbacks=[tr_eval], max_epochs=config["epochs"], logger=wandb_logger)
    trainer.fit(model, datamodule=dm)

train()
