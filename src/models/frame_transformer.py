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
        # with torch.no_grad():
        representations = self.backbone(x)
        return representations


class FrameTransformer(pl.LightningModule):
    def __init__(self, **kwargs):
        super(FrameTransformer, self).__init__()
        self.save_hyperparameters()
        if self.hparams.cls:
            self.hparams.seq_len += 1
        self.criterion = nn.BCEWithLogitsLoss()
        self.position_encoder = PositionalEncoding(
            896, 0.5,
            max_len=50)
        self.img_model = ImgResNet()
        # self.cls_token = nn.Parameter(
        #     torch.randn(1, 1, self.hparams.input_dimension))
        self.scene_transformer = TransformerBase(896, 896, 8, 896, 8, 0.6)
        self.running_labels = []
        self.running_logits = []
        self.img_cls = nn.Parameter(torch.rand(1, 3, 224, 224))
        self.mlp_head = nn.Sequential(nn.LayerNorm(896), nn.Linear(896, 15))
        self.decoder = nn.Sequential(nn.Linear(75, 32), nn.GELU(), nn.Dropout(
            0.5), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 15))
        self.encoder = nn.Sequential(nn.Linear(256, 256), nn.Dropout(0.5))
        self.running_logits = []
        self.running_labels = []
        self.val_auroc = AUROC(num_classes=15)
        self.train_auroc = AUROC(num_classes=15)
        self.train_aprc = AveragePrecision(num_classes=15)
        self.norm = nn.LayerNorm(896)
        self.tpn = TPN()
        self.val_aprc = AveragePrecision(num_classes=15)
        self.pool = nn.AdaptiveAvgPool2d((1, 15))

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
        #                            momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)

        optimizer = torch.optim.Adam(self.parameters(
        ), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        # optimizer = torch.optim.Adagrad(self.parameters(
        # ), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def forward(self, data):
        total = []
        # d = data.squeeze(0)
        # img_embedding = self.img_model(d) #imgs channel width height
        # data = [2, 50, 3, 224, 224]
        # cls = [2, 50, 3, 224, 224]
        for d in range(len(data)):
            cls_d = torch.cat((data[d], self.img_cls), dim=0)
            total.append(cls_d)
        data = torch.stack(total)
        data = data.view(-1, 3, 224, 224)
        data = self.img_model(data)
        # data = [batch + cls, dim]
        data = data.view(self.hparams.batch_size, self.hparams.seq_len, -1)
        data = data.permute(1, 0, 2)
        data = self.position_encoder(data)
        img_seq = data.permute(1, 0, 2)
        img_seq = self.norm(img_seq)
        img_seq = img_seq.permute(1, 0, 2)
        img_seq = self.scene_transformer(data)
        img_seq = img_seq.permute(1, 0, 2)
        output = img_seq[:, 0]
        output = self.mlp_head(output)

        return output

    def training_step(self, batch, batch_idx):
        target, data = batch
        data = data.float()
        # target = self.label_tidy(target)
        # save_image(grid, "test.png")
        data = self(data)
        target = target.float()
        loss = self.criterion(data, target)
        target = target.int()
        self.train_auroc(data, target)
        self.train_aprc(data, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/auroc", self.train_auroc, on_step=True, on_epoch=True)
        self.log("train/aprc", self.train_aprc, on_step=True, on_epoch=True)
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
        target, data = batch
        data = data.float()
        # target = self.label_tidy(target)
        grid = make_grid(data[0], nrow=10)
        data = self(data)
        target = target.float()
        loss = self.criterion(data, target)
        target = target.int()
        sig_data = F.sigmoid(data)
        self.running_logits.append(sig_data)
        self.running_labels.append(target)
        format_target = self.translate_labels(target[0])
        format_logits = self.translate_labels((sig_data[0] > 0.2).to(int))
        images = wandb.Image(
            grid, caption=f"predicted: {format_logits}, actual {format_target}")
        self.logger.experiment.log({"examples": images})
        self.val_auroc(data, target)
        self.val_aprc(data, target)
        # self.val_f1_2(data, target)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_step=True, on_epoch=True)
        self.log("val/aprc", self.val_aprc, on_step=True, on_epoch=True)
        return loss
