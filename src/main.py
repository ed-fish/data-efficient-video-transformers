import pytorch_lightning as pl
import torch.nn as nn
import wandb
import yaml
import torch
from pytorch_lightning.loggers import WandbLogger
from dataloaders.mmx.MMX_Temporal_dl import MMXDataModule
from dataloaders.mmx.MMX_Frame_dl import MMXFrameDataset, MMXFrameDataModule
from dataloaders.mmx.MMX_Light_dl import MMXLightDataset, MMXLightDataModule
from dataloaders.mit.MIT_Temporal_dl import MITDataModule, MITDataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.LSTM import LSTMRegressor
from models.transformer import SimpleTransformer
from models.frame_transformer import FrameTransformer
from callbacks.callbacks import TransformerEval, DisplayResults, MITEval
from yaml.loader import SafeLoader
import torch.nn as nn

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

if __name__ == "__main__":
    torch.manual_seed(1130)
    callbacks = []
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
    wandb.init(project="transformer-frame-video", name="seed_test_1130",
               config=data)
    config = wandb.config
    print(config["data_set"])

    wandb_logger = WandbLogger(
        project=config["logger"])

    if config["model"] == "ptn" or config["model"] == "ptn_shared":
        model = SimpleTransformer(**config)
    elif config["model"] == "lstm":
        model = LSTMRegressor(seq_len=200, batch_size=64,
                              criterion=nn.BCELoss(), n_features=4608, hidden_size=512, num_layers=4,
                              dropout=0.2, learning_rate=0.00005)
    elif config["model"] == "frame_transformer" or config["model"] == "distil" or config["model"] == "sum" or config["model"] == "frame" or config["model"] == "vid" or config["model"] == "pre_modal" or config["model"] == "sum_residual":
        model = FrameTransformer(**config)

    if config["data_set"] == "mit":
        miteval = MITEval()
        dm = MITDataModule("data/mit/MIT_train_temporal.pkl",
                           "data/mit/MIT_validation_temporal.pkl", config)
        callbacks = [miteval]

    elif config["data_set"] == "mmx":
        transformer_callback = TransformerEval()
        dm = MMXDataModule("data/mmx/mmx_train_temporal.pkl",
                           "data/mmx/mmx_val_temporal.pkl", config)
        callbacks = [transformer_callback]
        # checkpoint = ModelCheckpoint(
        #     save_top_k=-1, dirpath="trained_models/mmx/double", filename="double-{epoch:02d}")
        # display = DisplayResults()

    elif config["data_set"] == "mmx-frame":
        transformer_callback = TransformerEval()
        dm = MMXLightDataModule("data/mmx/light/out.csv", config)
        if config["test"]:
            display = DisplayResults()
            callbacks = [transformer_callback]
        else:
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
    #trainer = pl.Trainer(gpus=1, logger=wandb_logger, callbacks=callbacks, accumulate_grad_batches=8, precision=16, max_epochs=50)

    trainer = pl.Trainer(gpus=1, logger=wandb_logger,
                         callbacks=callbacks, max_epochs=1000)
    model = model.load_from_checkpoint("transformer-frame-video/2wxq6ed1/checkpoints/epoch=32-step=24947.ckpt")

    #trainer.fit(model, datamodule=dm)
    # dm.setup()
    # loader = dm.val_dataloader()
    # target, img, vid, frame_list = next(iter(loader))
    # model = model.vid_model.backbone
    # target_layers = [model.layer4[-1]]

    # vid = vid.view(-1, 12, 3,  112, 112)
    # vid = vid.permute(0, 2, 1, 3, 4)
    # vid = vid[0].unsqueeze(0)
    # print(vid.shape)

    # cam = GradCAM(model=model, target_layers=target_layers)
    # grayscale_cam = cam(input_tensor=vid)
    # print(grayscale_cam.shape)
    # grayscale_cam = grayscale_cam[0, :]
    # print(grayscale_cam.shape)
    # visualization = show_cam_on_image(frame_list[0], grayscale_cam, use_rgb=True)


    trainer.test(model, datamodule=dm, ckpt_path="transformer-frame-video/2wxq6ed1/checkpoints/epoch=32-step=24947.ckpt")
