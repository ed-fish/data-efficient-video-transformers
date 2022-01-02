import pytorch_lightning as pl
import torch.nn as nn
import wandb
import yaml
import torch
from pytorch_lightning.loggers import WandbLogger
from dataloaders.mmx.MMX_Temporal_dl import MMXDataModule
from dataloaders.mmx.MMX_Frame_dl import MMXFrameDataset, MMXFrameDataModule
from dataloaders.mit.MIT_Temporal_dl import MITDataModule, MITDataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.LSTM import LSTMRegressor
from models.transformer import SimpleTransformer
from models.frame_transformer import FrameTransformer
from callbacks.callbacks import TransformerEval, DisplayResults, MITEval
from yaml.loader import SafeLoader
import torch.nn as nn


def main():
    callbacks = []
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
    wandb.init(project="transformer-frame-video", name="mmx-frame-test",
               config=data)
    config = wandb.config
    print(config["data_set"])

    wandb_logger = WandbLogger(
        project=config["logger"])

    if config["model"] == "simple_transformer":
        model = SimpleTransformer(**config)
    elif config["model"] == "lstm":
        model = LSTMRegressor(seq_len=200, batch_size=64,
                              criterion=nn.BCELoss(), n_features=4608, hidden_size=512, num_layers=4,
                              dropout=0.2, learning_rate=0.00005)
    elif config["model"] == "frame_transformer":
        model = FrameTransformer(**config)

    if config["data_set"] == "mit":
        miteval = MITEval()
        dm = MITDataModule("data/mit/MIT_train_temporal.pkl",
                           "data/mit/MIT_validation_temporal.pkl", config)

        callbacks = [miteval]

    elif config["data_set"] == "mmx":
        transformer_callback = TransformerEval()
        dm = MMXDataModule("data/mmx/temporal/mmx_train_temporal.pkl",
                           "data/mmx/temporal/mmx_val_temporal.pkl", config)
        callbacks = [transformer_callback]
        # checkpoint = ModelCheckpoint(
        #     save_top_k=-1, dirpath="trained_models/mmx/double", filename="double-{epoch:02d}")
        # display = DisplayResults()

    elif config["data_set"] == "mmx-frame":
        transformer_callback = TransformerEval()
        dm = MMXFrameDataModule("data/mmx/frames/mmx_train_temporal_frames.pkl",
                                "data/mmx/frames/mmx_val_temporal_frames.pkl", config)
        callbacks = [transformer_callback]

    else:
        assert(
            "No dataset selected, please update the configuration \n mit, mmx, mmx-frame")

    def weights_init_normal(m):
        '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0, 1/np.sqrt(y))
        # m.bias.data should be 0
            m.bias.data.fill_(0)

    # weights_init_normal(model)

    trainer = pl.Trainer(gpus=[1], callbacks=callbacks,
                         logger=wandb_logger, gradient_clip_val=0.9, max_epochs=2000)
    trainer.fit(model, datamodule=dm)
    # model = model.load_from_checkpoint(
    #     "trained_models/mmx/double/double-epoch=127-v1.ckpt", **config)
    # trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main()
