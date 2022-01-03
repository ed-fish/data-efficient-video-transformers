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
        if self.hparams.cls:
            self.hparams.seq_len += 1
        self.criterion = nn.BCEWithLogitsLoss()
        self.position_encoder = PositionalEncoding(2048, self.hparams.dropout,
            max_len=self.hparams.seq_len)

        self.encoder_layers0 = TransformerEncoderLayer(
            self.hparams.input_dimension, self.hparams.nhead, self.hparams.nhid, self.hparams.dropout)
        self.transformer_encoder0 = TransformerEncoder(
            self.encoder_layers0, self.hparams.nlayers)

        self.encoder_layers1 = TransformerEncoderLayer(
            self.hparams.input_dimension, self.hparams.nhead, self.hparams.nhid, self.hparams.dropout)
        self.transformer_encoder1 = TransformerEncoder(
            self.encoder_layers1, self.hparams.nlayers)

        self.norm = nn.LayerNorm(2048)
        self.running_labels = []
        self.running_logits = []
        self.cls = nn.Parameter(torch.rand(
            1, self.hparams.batch_size, 2048))
        self.mlp_head = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 15))
        self.mlp_encoder = nn.Sequential(
            nn.LayerNorm(2048), nn.Linear(2048, 1024))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                                    momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)

        # optimizer = torch.optim.AdamW(self.parameters(
        # ), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def forward(self, src):
        src = self.expert_encoder(src)
        src = src * math.sqrt(self.hparams.input_dimension//2)
        src = self.position_encoder(src)
        # src = self.norm(src)
        output = self.transformer_encoder(src)
        return output

    def add_pos_cls(self, data):
        # input is expert, batch, sequence, dimension
        data = rearrange(data, 'b s d -> s b d')
        data = torch.cat((self.cls, data))
        data = self.position_encoder(data)
        data = rearrange(data, 's b d -> b s d')
        data = self.norm(data)
        data = rearrange(data, 'b s d -> s b d')
        return data

    def ptn_shared(self, data):
        # data input (BATCH, SEQ, EXPERTS, DIM)
        # experts batch sequence dimension
        expert_array = []
        data = rearrange(data, 'b s e d -> e b s d')
        for expert in data:
            # experts sequence batch dimension
            e = self.add_pos_cls(expert)  # s b d (s > seq len)
            e = self(e)
            e = rearrange(e, 's b d ->  b s d')
            e = e[:, 0]
            print("e", e.shape)
            expert_array.append(e)
        expert_array = torch.stack(expert_array)  # elen b d
        expert_array = rearrange(expert_array, 's b d -> b s d')
        expert_array = self.add_pos_cls(expert_array)
        ptn_out = self(expert_array)
        ptn_out = rearrange(ptn_out, 's b d ->  b s d')
        ptn_out = ptn_out[:, 0]
        ptn_out = self.mlp_head(ptn_out)
        return ptn_out

    def ptn(self, data):
        # data input (BATCH, SEQ, EXPERTS, DIM)
        # experts batch sequence dimension
        expert_array = []
        data = rearrange(data, 'b s e d -> e b s d')
        for i, expert in enumerate(data):
            # experts sequence batch dimension
            e = self.add_pos_cls(expert)  # s b d (s > seq len)
            print("e", e.shape)
            if i == 0:
                e = self.transformer_encoder0(e)
            elif i == 1:
                e = self.transformer_encoder1(e)
            print("e1", e.shape)
            e = rearrange(e, 's b d ->  b s d')

            print("e2", e.shape)
            e = e[:, 0, :]
            print("e3", e.shape)
            expert_array.append(e)

        ptn_out = torch.stack(expert_array)  # elen b d
        ptn_out = rearrange(ptn_out, "e b d -> b e d")
        print("ptn", ptn_out.shape)
        ptn_out = torch.sum(ptn_out, dim=1)
        print("ptn out", ptn_out.shape)
        ptn_out = self.mlp_head(ptn_out)
        return ptn_out

    def training_step(self, batch, batch_idx):

        data = batch["experts"]
        target = batch["label"]

        data = self.shared_step(data)
        #target = self.format_target(target)
        loss = self.criterion(data, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch["experts"]
        target = batch["label"]
        data = self.shared_step(data)
        #target = self.format_target(target)
        loss = self.criterion(data, target)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        target = target.int()
        sig_data = F.sigmoid(data)
        self.running_logits.append(sig_data)

        self.running_labels.append(target)
        self.running_logits.append(data)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        return loss

    def shared_step(self, data):
        if self.hparams.model == "ptn":
            data = self.ptn(data)
            return data
        if self.hparams.model == "ptn_shared":
            data = self.ptn(data)
            return data


    def format_target(self, target):
        target = torch.cat(target, dim=0)
        target = target.squeeze()
        return target

