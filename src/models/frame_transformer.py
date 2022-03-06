import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from torchmetrics import AUROC, F1, AveragePrecision
from einops import rearrange
import wandb
from models import custom_resnet
from torchvision.utils import make_grid, save_image
import pickle as pkl
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_model, dropout=0.1, max_len=4):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBase(pl.LightningModule):
    def __init__(self, input_dimension, output_dimension, nhead, nhid,
                 nlayers, dropout):
        super(TransformerBase, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            input_dimension, nhead, nhid, dropout)
        nlayers = nlayers
        self.transformer = TransformerEncoder(encoder_layer, nlayers)

    def forward(self, x):
        return self.transformer(x)


class ImgResNet(pl.LightningModule):
    def __init__(self):
        super(ImgResNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_filters = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(num_filters, 896))

    def forward(self, x):
        # self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.backbone(x)
            return representations


class VidResNet(pl.LightningModule):
    def __init__(self):
        super(VidResNet, self).__init__()
        self.backbone = models.video.r2plus1d_18(pretrained=True)
        num_filters = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(num_filters, 896))

    def forward(self, x):
        #with torch.no_grad():
        representations = self.backbone(x)
        return representations


# class LocationResNet(pl.LightningDataModule):
#     def __init__(self):
#         super(LocationResNet, self).__init__()
#         self.backbone =


class FrameTransformer(pl.LightningModule):
    def __init__(self, **kwargs):
        super(FrameTransformer, self).__init__()
        self.save_hyperparameters()
        if self.hparams.cls:
            self.hparams.seq_len += 1
        self.criterion = nn.BCEWithLogitsLoss()
        self.distil_criterion = nn.CrossEntropyLoss()
        self.position_encoder = PositionalEncoding(
            896, 0.5,
            max_len=14)
        #self.img_model = ImgResNet()
        self.vid_model = VidResNet()
        # self.cls_token = nn.Parameter(
        #     torch.randn(1, 1, self.hparams.input_dimension))
        #self.scene_transformer = TransformerBase(896, 896, 4, 896, 4, 0.5)
        self.distil_transformer = TransformerBase(896, 128, 2, 512, 4, 0.5)
        self.running_labels = []
        self.running_logits = []
        self.running_paths = []
        self.running_embeds = []
        #self.img_cls = nn.Parameter(torch.rand(1, 3, 224, 224))
        self.vid_cls = nn.Parameter(torch.rand(1, 12, 3, 112, 112))
        self.img_mlp_head = nn.Sequential(nn.Linear(896, 512), nn.GELU(), nn.Linear(512, 128),nn.GELU(), nn.Linear(128, 19))
        #self.vid_mlp_head = nn.Sequential(
            #nn.LayerNorm(896), nn.Linear(896, 19))
        # self.decoder = nn.Sequential(nn.Linear(75, 32), nn.GELU(), nn.Dropout(
        # 0.5), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 15))
        # self.encoder = nn.Sequential(nn.Linear(256, 256), nn.Dropout(0.5))
        self.running_logits = []
        self.running_labels = []
        #self.val_auroc = AUROC(num_classes=19)
        #self.train_auroc = AUROC(num_classes=19)
        self.train_aprc = AveragePrecision(num_classes=19)
        self.norm = nn.LayerNorm(896)
        # self.tpn = TPN()
        self.val_aprc = AveragePrecision(num_classes=19)
        # self.pool = nn.AdaptiveAvgPool2d((1, 15))
        self.cos = nn.CosineSimilarity(dim=1)

    def configure_optimizers(self):
        if self.hparams.opt == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                                        momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        elif self.hparams.opt == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(
            ), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        elif self.hparams.opt == "adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(
            ), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def forward(self, img, vid):
        total = []
        # out_img = self.img_step(img)
        if self.hparams.model == "distil":
            img, vid = self.distillation_step(img, vid)
            return img, vid

        if self.hparams.model == "sum":
            img, vid = self.distillation_step(img, vid)
            output = img + vid
            output = self.img_mlp_head(output)
            return output

        if self.hparams.model == "sum_residual":
            vid_cls = self.vid_step(vid)
            img_cls, seq = self.img_step(img, vid_cls)
            embed_list = []
            # for s in seq:
            #     s = self.img_mlp_head(s)
            #     embed_list.append(s)
            # return embed_list
            img_cls = F.normalize(img_cls, p=2.0, dim=-1)
            vid_cls = F.normalize(img_cls, p=2.0, dim=-1)
            embed = img_cls + vid_cls
            output = self.img_mlp_head(embed)
            return output

        if self.hparams.model == "post_sum":
            img, vid, vid_cls = self.distillation_step(img, vid)
            output = img + vid_cls
            output = self.img_mlp_head(output)
            return output

        if self.hparams.model == "frame":
            img_cls = self.img_step(img, None)
            return img_cls

        if self.hparams.model == "pre_modal":
            img_cls = self.pre_modal(img, vid)
            return img_cls

        if self.hparams.model == "vid":
            vid_cls = self.vid_step(vid)
            vid_cls = self.img_mlp_head(vid_cls)
            return vid_cls

    def distillation_step(self, img, vid):
        vid_cls = self.vid_step(vid)
        img_cls, vid_tkn = self.img_step(img, vid_cls)
        return img_cls, vid_tkn

    def pre_modal(self, img, vid):
        vid = self.vid_step
        img_cls = self.img_step(img, vid)
        return img_cls

    def vid_step(self, data):
        total = []
        for d in range(len(data)):
            cls_d = torch.cat((self.vid_cls, data[d]), dim=0)
            total.append(cls_d)
        data = torch.stack(total)
        data = data.view(-1, 12, 3,  112, 112)
        data = data.permute(0, 2, 1, 3, 4)
        data = self.vid_model(data)

        if self.hparams.model == "pre-modal":
            return data
        data = data.view(self.hparams.batch_size, 14, 896)
        data = data.permute(1, 0, 2)
        data = self.position_encoder(data)
        data = self.distil_transformer(data)
        data = data.permute(1, 0, 2)
        vid_cls = data[:, 0]
        return vid_cls

    def img_step(self, data, distil_inject):
        total = []
        for d in range(len(data)):
            cls_d = torch.cat((self.img_cls, data[d]), dim=0)
            total.append(cls_d)
        data = torch.stack(total)
        data = data.view(-1, 3, 224, 224)
        data = self.img_model(data)
        if self.hparams.model == "pre-modal":
            data = data + distil_inject
        # data = [batch + cls, dim]
        data = data.view(self.hparams.batch_size, self.hparams.seq_len, -1)
        data = data.permute(1, 0, 2)
        if self.hparams.model == "sum":
            data = torch.cat((data, distil_inject))
        data = self.position_encoder(data)
        #img_seq = data.permute(1, 0, 2)
        #img_seq = self.norm(img_seq)
        #img_seq = img_seq.permute(1, 0, 2)
        img_seq = self.scene_transformer(data)
        img_seq = img_seq.permute(1, 0, 2)
        cls = img_seq[:, 0]
        if self.hparams.model == "distil":
            dis_tkn = img_seq[:, -1]
            return cls, dis_tkn
        if self.hparams.model == "sum":
            dis_tkn = img_seq[:, -1]
            return cls, dis_tkn
        if self.hparams.model == "sum_residual":
            return cls, img_seq
        else:
            cls = self.img_mlp_head(cls)
            return cls

    def training_step(self, batch, batch_idx):
        if self.hparams.model == "distil":
            target, img, vid = batch
            img, vid = self(img, vid)
            distil_loss = self.distil_criterion(img, torch.argmax(vid, dim=-1))
            base_loss = self.criterion(img, target)
            loss = base_loss + distil_loss
            self.log("train/distilloss", distil_loss,
                     on_step=True, on_epoch=True)
            self.log("train/bass_loss", base_loss,
                     on_step=True, on_epoch=True)
            self.log("train/cossim", self.cos(img, vid)[0],
                     on_step=True, on_epoch=True)
            data = img
        if self.hparams.model == "sum" or self.hparams.model == "pre_modal" or self.hparams.model == "sum_residual":
            target, img, vid = batch
            data = self(img, vid)
            loss = self.criterion(data, target)
        if self.hparams.model == "frame":
            target, img, vid = batch
            data = self(img, None)
            target = target.float()
            loss = self.criterion(data, target)
        if self.hparams.model == "vid":
            target, img, vid = batch
            data = self(None, vid)
            target = target.float()
            loss = self.criterion(data, target)

        target = target.int()
        #self.train_auroc(data, target)
        self.train_aprc(data, target)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        
        #self.log("train/auroc", self.train_auroc, on_step=True, on_epoch=True)
        self.log("train/aprc", self.train_aprc, on_step=False, on_epoch=True)
        return loss

    def translate_labels(self, label_vec):
        target_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                        'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Science Fiction', 'Thriller',  'War']
        labels = []
        for i, l in enumerate(label_vec):
            if l:
                labels.append(target_names[i])
        return labels

    def validation_step(self, batch, batch_idx):
        if self.hparams.model == "distil":
            img, vid = self(img, vid)
            distil_loss = self.distil_criterion(img, torch.argmax(vid, dim=-1))
            base_loss = self.criterion(img, target)
            self.log("val/distilloss", distil_loss,
                     on_step=True, on_epoch=True)
            self.log("val/base_loss", base_loss, on_step=True, on_epoch=True)
            self.log("val/cossim", self.cos(img, vid)[0],
                     on_step=True, on_epoch=True)
            loss = base_loss + distil_loss
            data = img
        elif self.hparams.model == "sum" or self.hparams.model == "pre_modal" or self.hparams.model == "sum_residual":
            target = batch[0]['label'].reshape(self.hparams.batch_size,-1,  19)
            target = target[:, 0, :]
            vid = batch[0]['data']
            vid = vid.reshape(self.hparams.batch_size, self.hparams.seq_len -1, self.hparams.frame_len, 3, 112, 112)

            data = self(img, vid)
            loss = self.criterion(data, target)
        elif self.hparams.model == "frame":
            target, img, vid = batch
            data = self(img, None)
            target = target.float()
            loss = self.criterion(data, target)
        elif self.hparams.model == "vid":
            target, img, vid = batch
            #target = batch[0]['label'].reshape(self.hparams.batch_size,-1,  19)
            #target = target[:, 0, :]
            #vid = batch[0]['data']
            #print(vid.shape)
            #print(target.shape)
            #vid = vid.reshape(self.hparams.batch_size, self.hparams.seq_len - 1, self.hparams.frame_len, 3, 112, 112)
            data = self(None, vid)
            target = target.float()
            loss = self.criterion(data, target)

        target = target.int()
        sig_data = F.sigmoid(data)
        self.running_logits.append(sig_data)
        self.running_labels.append(target)
        #format_target = self.translate_labels(target[0])
        #format_logits = self.translate_labels((sig_data[0] > 0.2).to(int))
        # images = wandb.Image(
        #     grid, caption=f"predicted: {format_logits}, actual {format_target}")
        # self.logger.experiment.log({"examples": images})
        #self.val_auroc(data, target)
        self.val_aprc(data, target)
        # self.val_f1_2(data, target)
        self.log("val/loss", loss, on_epoch=True)
        #self.log("val/auroc", self.val_auroc, on_step=True, on_epoch=True)
        self.log("val/aprc", self.val_aprc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.hparams.model == "distil":
            img, vid, path = self(img, vid)
            data = img
        elif self.hparams.model == "sum" or self.hparams.model == "pre_modal" or self.hparams.model == "sum_residual":
            target, img, vid = batch
            embed = self(img, vid)
        elif self.hparams.model == "frame":
            target, img, path = batch
            data = self(img, None)
            target = target.float()
        elif self.hparams.model == "vid":
            target, img, vid = batch
            data = self(None, vid)
            target = target.float()

        target = target.int()
        self.running_logits.append(F.sigmoid(data))
        # self.running_embeds.append(data)
        self.running_labels.append(target)
        # self.running_paths.append(path)

