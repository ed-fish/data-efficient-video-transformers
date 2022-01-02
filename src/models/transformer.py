import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SimpleTransformer(pl.LightningModule):
    def __init__(self, **kwargs):
        super(SimpleTransformer, self).__init__()

        self.save_hyperparameters()
        self.expert_encoder = nn.Sequential(
            nn.Linear(self.hparams.input_dimension,
                      self.hparams.input_dimension//2),
        )
        if self.hparams.cls:
            self.hparams.seq_len += 1
        self.criterion = nn.CrossEntropyLoss()
        self.position_encoder = PositionalEncoding(
            self.hparams.input_dimension, self.hparams.dropout,
            max_len=self.hparams.seq_len)
        self.encoder_layers = TransformerEncoderLayer(
            self.hparams.input_dimension, self.hparams.nhead, self.hparams.nhid, self.hparams.dropout)
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layers, self.hparams.nlayers)
        self.norm = nn.LayerNorm(self.hparams.input_dimension)
        # self.cls_token = nn.Parameter(
        #     torch.randn(1, 1, self.hparams.input_dimension))

        post_encoder_layer = TransformerEncoderLayer(
            self.hparams.token_embedding, 1, self.hparams.nhid, self.hparams.dropout)
        self.post_transformer_encoder = TransformerEncoder(
            post_encoder_layer, self.hparams.nlayers)
        self.running_labels = []
        self.running_logits = []
        self.classifier = nn.Sequential(
            nn.Linear(self.hparams.input_dimension//2 * self.hparams.seq_len,
                      self.hparams.input_dimension),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.input_dimension,
                      self.hparams.input_dimension // 2),
            nn.ReLU(),
            nn.Linear(self.hparams.input_dimension //
                      2, self.hparams.n_classes),
        )
        self.cat_classifier = nn.Sequential(
            nn.Linear(len(self.hparams.experts) * 305, 305)
        )
        self.cls = nn.Parameter(torch.rand(
            self.hparams.seq_len, self.hparams.batch_size, 2048))

        self.mlp_head = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 15))

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
        #                             momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)

        optimizer = torch.optim.AdamW(self.parameters(
        ), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def forward(self, src):
        output = self.transformer_encoder(src)
        return output

    def add_pos_cls(self, data):
        # input is expert, batch, sequence, dimension
        data = self.position_encoder(data)
        data = torch.cat((self.cls, data))
        return data

    def ptn(self, data):
        # data input (BATCH, SEQ, EXPERTS, DIM)
        # experts batch sequence dimension
        expert_array = []
        data = rearrange(data, 'b s e d -> e b s d')
        for expert in data:
            # experts sequence batch dimension
            e = rearrange(expert, 'b s d -> s b d')
            e = self.add_pos_cls(e)  # s b d (s > seq len)
            e = self(e)
            e = e[0]
            expert_array.append(e)
        expert_array = torch.stack(expert_array)  # elen b d
        expert_array = self.add_pos_cls(expert_array)
        ptn_out = self(expert_array)
        ptn_out = ptn_out[0]
        ptn_out = self.mlp_head(ptn_out)
        return ptn_out

    def training_step(self, batch, batch_idx):
        data = batch["experts"]
        target = batch["label"]
        data = self.ptn(data)
        #target = self.format_target(target)
        loss = self.criterion(data, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch["experts"]
        target = batch["label"]
        data = self.ptn(data)
        #target = self.format_target(target)
        loss = self.criterion(data, target)
        target = target.int()
        sig_data = F.sigmoid(data)
        self.running_logits.append(sig_data)
        self.running_labels.append(target)
        return loss

    def format_target(self, target):
        target = torch.cat(target, dim=0)
        target = target.squeeze()
        return target
