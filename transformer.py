import torch
import confuse
import torch.nn as nn
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from dataloaders.MIT_Temporal_dl import MITDataset, MITDataModule
import math
import pytorch_lightning as pl

class PositionalEncoding(nn.Module):
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


class TransformerModel(pl.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

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

        #src = self.encoder(src) * math.sqrt(self.ninp)
        #src = self.encoder(src) * math.sqrt(self.ninp)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = F.softmax(output)
        return output


# Dataloading
# The model expects shape of Sequence, Batch, Embedding
# For MIT the batches should be [3, 32, 1028] the total size of the vocabulary 

##### Training ####


def train():

    ntokens = 305
    emsize = 2048
    nhid = 2048
    nlayers = 4
    nhead = 4
    dropout = 0.3
    device = torch.device("cuda:2")
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.25
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    config = confuse.Configuration("mmodel-moments-in-time")
    config.set_file("config.yaml")
    dm = MITDataModule("data/mit/mit_tensors_train_wc.pkl","data/mit/mit_tensors_train_wc.pkl", config)
    dm.setup(stage="fit")
    data_loader = dm.train_dataloader()
    bptt = config["batch_size"].get()

    model.train()
    total_loss = 0

    src_mask = model.generate_square_subsequent_mask(3).to(device)
    for epoch in range(10000):
        for idx, batch in enumerate(data_loader):
            data = batch["experts"]
            target = batch["label"]

            data = torch.cat(data, dim=0)
            target = torch.cat(target, dim=0)
            target = target.to(device)

            # data must be transposed to sequenceBC from BSC
            data = data.permute(1, 0, 2)
            #print("after permute for SBE", data.shape)
            #print(data[0])
            
            data = data.to(device)

            #target = target.to(device)
            optimizer.zero_grad()
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = model(data, src_mask)
            #print("output_shape", output.shape)

            # will reshape to [96, 305]
            transform_t = output.view(-1, ntokens)

            # options for classification
            # average pooling + softmax

            transform_t = transform_t.unsqueeze(0)
            pooled_result = F.adaptive_avg_pool2d(transform_t, (64, 305))
            pooled_result = pooled_result.squeeze(0)

            # Cross Entropy includes softmax https://bit.ly/3f73RJ7
            # pooled_result = F.log_softmax(pooled_result, dim=-1)
            
            target = target.squeeze()
            #print("prediction", torch.argmax(pooled_result, dim =1))
            #print("target", target)
            
            loss = criterion(pooled_result, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            log_interval = 10
            if idx % log_interval == 0 and idx > 0:
                curr_loss = total_loss / log_interval
                print(f"epoch {epoch},step {idx}, loss {total_loss}")
            total_loss = 0
        scheduler.step()

train()

