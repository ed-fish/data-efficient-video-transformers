import confuse 
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional import f1, auroc

from dataloaders.MIT_Temporal_dl import MITDataset, MITDataModule
from dataloaders.MMX_Temporal_dl import MMXDataset, MMXDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_model, dropout=0.1, max_len=5):
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

    def __init__(self, ntoken, ninp, nhead=4, nhid=2048, nlayers=4, learning_rate=0.05, dropout=0.5, warmup_epochs=10, max_epochs=100, seq_len=5):
        super(TransformerModel, self).__init__()

        self.save_hyperparameters('nhead','nhid','learning_rate','dropout', 'warmup_epochs', 'max_epochs')

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.seq_len = seq_len

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=seq_len) # shared dropout value for pe and tm(el)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp).to(self.device)
        #self.accuracy = torchmetrics.F1(num_classes=15, threshold=0.1, top_k=3) # f1 weighted for mmx
        #self.accuracy = torchmetrics.Accuracy()
        self.decoder = nn.Linear(ninp, 128)
        self.classifier = nn.Linear(128 * seq_len, ntoken)
        self.init_weights()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=0,
            eta_min=0
            )

        scheduler = {
            'name': "warmup cosine decay",
            'scheduler': linear_warmup_cosine_decay,
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]

#        return optimizer

    def shared_step(self, data, target):

        data = torch.cat(data, dim=0)
        target = torch.cat(target, dim=0)

        # Permute to [ S, B, E ]
        data = data.permute(1, 0, 2)
        # print("after permute for SBE", data.shape)
        # print(data[0])

        # just for cases where drop_last==False
        # ensures mask is same as batch size
        #if data.size(0) != bptt:
        src_mask = self.generate_square_subsequent_mask(data.size(0))
        src_mask = src_mask.to(self.device)

        output = self(data, src_mask)

        # will reshape to [96, 305]
        transform_t = output.permute(1, 0, 2)
        #transform_t = output.view(32, -1)

        # options for classification
        # average pooling + softmax

        print(transform_t.shape)
        transform_t = transform_t.reshape(32, -1)
        print(transform_t.shape)
        pooled_result = self.classifier(transform_t)

        # pooled_result = F.adaptive_avg_pool3d(transform_t, (32, 1, 15))
        # pooled_result = pooled_result.squeeze().squeeze()

        # Cross Entropy includes softmax https://bit.ly/3f73RJ7
        # pooled_result = F.log_softmax(pooled_result, dim=-1)

        target = target.squeeze()

        return pooled_result, target


    def training_step(self, batch, batch_idx):
        data = batch["experts"]
        target = batch["label"]

        data, target = self.shared_step(data, target)
        #print("guess", torch.argmax(F.log_softmax(data), dim=-1))
        #print("target", target)

        # Concat sequence of data [ B, S, E ]
        # print("prediction", torch.argmax(pooled_result, dim =1))

        # target = torch.argmax(target, dim=-1)
        target = target.float()

        print(data[0])
        print(target[0])

        loss = self.criterion(data, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        #acc_preds = self.preds_acc(data)
        self.log('train_acc_step', f1(data, target.to(int), num_classes=15, threshold=-1.0))
        # torch.nn.utils.clip_grad_norm(self.parameters(), 0.5)

        return loss

    def preds_acc(self, preds):
        outputs = torch.sigmoid(preds)  # torch.Size([N, C]) e.g. tensor([[0., 0.5, 0.]])
        outputs[outputs >= 0.5] = 1
        return outputs

    def validation_step(self, batch, batch_idx):

        data = batch["experts"]
        target = batch["label"]

        data, target = self.shared_step(data, target)
        target = target.float()

        #target = torch.argmax(target, dim=-1)
        loss = self.criterion(data, target)

        # acc_preds = self.preds_acc(data)

        self.log('val_acc_step_1', f1(data, target.to(int), num_classes=15, average="weighted",threshold=-1.0))
        self.log('val_acc_step_1.5', f1(data, target.to(int), num_classes=15,average="weighted", threshold=-1.5))
        self.log('val_acc_step_2', f1(data, target.to(int), num_classes=15,average="weighted", threshold=-2.0))
        self.log('val_acc_step_0', f1(data, target.to(int), num_classes=15,average="weighted", threshold=-0))
        self.log('auprc', auroc(data, target.to(int), num_classes=15,average="weighted"))
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

        src_mask = self.generate_square_subsequent_mask(self.seq_len)
        src = self.pos_encoder(src)
        src_mask = src_mask.to(self.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        # output = F.softmax(output)
        # Do not include softmax if nn.crossentropy as softmax included via NLLoss
        return output


# Dataloading
# The model expects shape of Sequence, Batch, Embedding
# For MIT the batches should be [3, 32, 1028] the total size of the vocabulary 

##### Training ####

def train():
    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("config.yaml")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # dm = MITDataModule("data/mit/mit_tensors_train_wc.pkl","data/mit/mit_tensors_train_wc.pkl", config)
    # dm = MMXDataModule("data/mmx/mmx_tensors_val.pkl","data/mmx/mmx_tensors_val.pkl", config)
    dm = MMXDataModule("data_processing/trailer_temporal/mmx_tensors_train.pkl", "data_processing/trailer_temporal/mmx_tensors_val.pkl", config)

    bptt = config["batch_size"].get()
    learning_rate = config["learning_rate"].get()
    epochs = config["epochs"].get()
    n_warm_up = int(epochs / 10)
    seq_len = config["seq_len"].get()
    ntokens = 15
    emsize = 512
    nhid = 128
    nlayers = 2
    nhead = 2
    dropout = 0.5
    model = TransformerModel(ntokens, emsize, nhead, nhid=nhid, nlayers=nlayers, learning_rate=learning_rate, dropout=dropout, warmup_epochs=n_warm_up, max_epochs=epochs, seq_len=seq_len)
    trainer = pl.Trainer(gpus=[config["gpu"].get()], callbacks=[lr_monitor], max_epochs=epochs)
    trainer.fit(model, datamodule=dm)

train()

