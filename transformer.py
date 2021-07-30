import torch
import confuse
import torch.nn as nn
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import torchmetrics
from dataloaders.MIT_Temporal_dl import MITDataset, MITDataModule
import math
import pytorch_lightning as pl

class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_model, dropout=0.1, max_len=3):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)        

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(pl.LightningModule):

    def __init__(self, ntoken, ninp, nhead=4, nhid=2048, nlayers=4, learning_rate=0.005, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.save_hyperparameters('nhead','nhid','learning_rate','dropout')

        self.criterion = nn.CrossEntropyLoss()

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp).to(self.device)
        self.accuracy = torchmetrics.Accuracy()
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def shared_step(self, data, target):

        data = torch.cat(data, dim=0)
        target = torch.cat(target, dim=0)

        # Permute to [ S, B, E ]
        data = data.permute(1, 0, 2)
        #print("after permute for SBE", data.shape)
        #print(data[0])

        # just for cases where drop_last==False
        # ensures mask is same as batch size
        #if data.size(0) != bptt:
        src_mask = self.generate_square_subsequent_mask(data.size(0))
        src_mask = src_mask.to(self.device)

        output = self(data, src_mask)
        #print("output_shape", output.shape)

        # will reshape to [96, 305]
        transform_t = output.view(-1, 305)

        # options for classification
        # average pooling + softmax

        transform_t = transform_t.unsqueeze(0)
        pooled_result = F.adaptive_avg_pool2d(transform_t, (32, 305))
        pooled_result = pooled_result.squeeze(0)

        # Cross Entropy includes softmax https://bit.ly/3f73RJ7
        # pooled_result = F.log_softmax(pooled_result, dim=-1)

        target = target.squeeze()

        return pooled_result, target


    def training_step(self, batch, batch_idx):
        data = batch["experts"]
        target = batch["label"]

        data, target = self.shared_step(data, target)

        # Concat sequence of data [ B, S, E ]
        #print("prediction", torch.argmax(pooled_result, dim =1))
        #print("target", target)

        self.log('train_acc_step', self.accuracy(torch.argmax(F.log_softmax(data), dim=-1), target))

        loss = self.criterion(data, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        torch.nn.utils.clip_grad_norm(self.parameters(), 0.5)

        return loss

    def validation_step(self, batch, batch_idx):

        data = batch["experts"]
        target = batch["label"]

        data, target = self.shared_step(data, target)
        loss = self.criterion(data, target)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        # SRC in NLP model referes to a vector representation of the word - this is learned while training the model. 
        # Options include:
        # - Just using the embeddings we have already
        # - Adding an additional linear layer
        # - Using bovw/knn and creating a sort of vocabulary from the expert embeddings

        #src = self.encoder(src) * math.sqrt(self.hparams.ninp)

        src_mask = self.generate_square_subsequent_mask(3)
        src = self.pos_encoder(src)
        src_mask = src_mask.to(self.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        #output = F.softmax(output)
        # Do not include softmax if nn.crossentropy as softmax included via NLLoss
        return output


# Dataloading
# The model expects shape of Sequence, Batch, Embedding
# For MIT the batches should be [3, 32, 1028] the total size of the vocabulary 

##### Training ####


def train():
    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("config.yaml")

    dm = MITDataModule("data/mit/mit_tensors_train_wc.pkl","data/mit/mit_tensors_train_wc.pkl", config)

    bptt = config["batch_size"].get()
    learning_rate = config["learning_rate"].get()
    ntokens = 305
    emsize = 2048
    nhid = 2048
    nlayers = 4
    nhead = 4
    dropout = 0.1
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
    trainer = pl.Trainer(gpus=[3])
    trainer.fit(model, datamodule=dm)


train()

