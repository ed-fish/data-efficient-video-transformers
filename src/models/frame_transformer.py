import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from einops import rearrange


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
        encoder_layer = TransformerEncoderLayer(input_dimension, nhead, nhid, dropout)
        nlayers = nlayers
        self.transformer = TransformerEncoder(encoder_layer, nlayers)
        
    def forward(self, x):
        return self.transformer(x)
    
 
class ImgResNet(pl.LightningModule):
    def __init__(self):
        super(ImgResNet, self).__init__()
        backbone = models.resnet18(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, 128)
        
    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
            x = self.classifier(representations)
            return x


class FrameTransformer(pl.LightningModule):
    def __init__(self, **kwargs):
        super(FrameTransformer, self).__init__()
        self.save_hyperparameters()
        if self.hparams.cls:
            self.hparams.seq_len += 1
        self.criterion = nn.BCEWithLogitsLoss()
        self.position_encoder = PositionalEncoding(
            128, 0.5,
            max_len=300)
        self.norm = nn.LayerNorm(self.hparams.input_dimension//2)
        self.img_model = ImgResNet()
        # self.cls_token = nn.Parameter(
        #     torch.randn(1, 1, self.hparams.input_dimension))
        
        self.img_transformer = TransformerBase(128, 128, 2, 128, 2, 0.5)
        self.clip_transformer = TransformerBase(128, 128, 2, 128, 2, 0.5)
        self.scene_transformer = TransformerBase(128, 128, 2, 128, 2, 0.5)
        self.running_labels = []
        self.running_logits = [] 
        img_cls = torch.rand(1, 128)
        self.register_buffer("img_cls", img_cls)
        self.mlp_head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 15))

    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
        #                            momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)

        optimizer = torch.optim.AdamW(self.parameters(
         ), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def forward(self, data):
        batch = []
        for b in data:  # batch
            scenes = [self.img_cls]
            for s in b:  # scenes
                clips = [self.img_cls]
                for c in s:  # clips
                    imgs = [self.img_cls]
                    for i in c:  # imgs
                        i = i.unsqueeze(0)
                        i = i.float()
                        img_embedding = self.img_model(i)
                        imgs.append(img_embedding)
                    imgs_stack = torch.stack(imgs)
                    pos_imgs = self.position_encoder(imgs_stack)
                    img_seq = self.img_transformer(pos_imgs)
                    clips.append(img_seq[0])
                clips_stack = torch.stack(clips)
                # clips_cls_stack = clips_stack[:, 0, :, :]
                pos_clips = self.position_encoder(clips_stack)
                clip_seq = self.clip_transformer(pos_clips)
                scenes.append(clip_seq[0])
            scenes_stack = torch.stack(scenes)
            scenes_pos = self.position_encoder(scenes_stack)
            output = self.scene_transformer(scenes_pos)
            batch.append(output[0])
        batch = torch.stack(batch)
        batch = batch.squeeze()
        return self.mlp_head(batch)

    def training_step(self, batch, batch_idx):
        target, data = batch
        #target = self.label_tidy(target)
        data = self(data)
        target = target.squeeze() 
        loss = self.criterion(data, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target, data = batch
        #target = self.label_tidy(target)
        data = self(data)
        target = target.squeeze()
        loss = self.criterion(data, target)
        _data = data.detach().cpu()
        _target = target.detach().cpu()
        _data = nn.Sigmoid()(_data)
        self.running_labels.append(_target)
        self.running_logits.append(_data)
        self.log("val/loss", loss, on_step=True, on_epoch=True, rank_zero_only=True)
        return loss
  