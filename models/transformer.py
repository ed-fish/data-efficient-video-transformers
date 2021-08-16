import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from dataloaders.MIT_Temporal_dl import MITDataset, MITDataModule
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, maxlen=5000):
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


class TransformerModel(nn.Module):

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

ntokens = 304
emsize = 512
nhid = 512
nlayers = 2
nhead = 2
dropout = 0.2
model = TransformerModer(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

import time

criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

config = confuse.Configuration("mmodel-moments-in-time")
config.set_file("config.yaml")
dm = MITDataModule("data_processing/mit_tensors_train_wc.pkl", config)
train_loader = dm.train_dataloader()



def train():
    model.train()
    total_loss = 0
    src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
    for batch, i in enumerate(train_loader):
        data = batch["x_i_experts"]
        target = batch["labels"]
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            curr_loss = total_loss / log_interval
            print(f"epoch {epoch}, loss {total_loss}")
        total_loss = 0

for epoch in range(1, 10):
    train()
    scheduler.step()




