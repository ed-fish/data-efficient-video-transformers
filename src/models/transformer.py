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
            self.hparams.input_dimension//2, self.hparams.dropout,
            max_len=self.hparams.seq_len)
        self.encoder_layers = TransformerEncoderLayer(
            self.hparams.input_dimension//2, self.hparams.nhead, self.hparams.nhid, self.hparams.dropout)
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layers, self.hparams.nlayers)
        self.norm = nn.LayerNorm(self.hparams.input_dimension//2)
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
 
    def post_transformer(self, data):
        data = torch.stack(data)
        data = data.transpose(0, 1) # b, expert, seq, data -> expert, b, seq, data
        if self.hparams.cls:
            collab_array = [torch.rand(1, 2048).to(self.device)]
        else:
            collab_array = []
        data = data.squeeze()
        
        for x in range(len(self.hparams.experts)):
            d = data[x, :, :, :]
            d = self.shared_step(d) # d = batch, seq, embeddings
            # d = rearrange(d, 'b s e -> s b e')
            collab_array.append(d)  
        stacked_array = torch.stack(collab_array)
        # stacked_array = self.pos_encoder(stacked_array)
        # stacked_array = stacked_array.transpose(0, 1)
        # src_mask = self.generate_square_subsequent_mask(stacked_array.size(0))
        # src_mask = src_mask.to(self.device)
        data = self.post_transformer_encoder(stacked_array)
        data = data.transpose(0, 1)
        data = data.reshape(self.hparams.batch_size, -1)
#        self.running_embeds.append(data)
        # output = self.decoder(data)

        transform_t = self.cat_classifier(data)
        pooled_result = transform_t.squeeze(0)
        return pooled_result

    def training_step(self, batch, batch_idx):

        data = batch["experts"]
        target = batch["label"]
        #data = self.shared_step(data)
        data = self.post_transformer(data)
        target = self.format_target(target)
        #target = target.float()
        loss = self.criterion(data, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch["experts"]
        target = batch["label"]
        # data = self.shared_step(data)
        data = self.post_transformer(data)
        target = self.format_target(target)
        # target = target.float()
        # target = torch.argmax(target, dim=-1)
        loss = self.criterion(data, target)
        # acc_preds = self.preds_acc(data)
        data = data.detach()
        data = F.softmax(data, dim=-1)
        data = torch.argmax(data, dim=-1)
        self.running_labels.append(target)
        self.running_logits.append(data)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        return loss

    def shared_step(self, data):
        # data = torch.stack(data)
        # data = torch.cat(data, dim=0)
        # data = data.squeeze()

        data = rearrange(data, 'b s e -> s b e')
        # FORWARD
        output = self(data)
        # print("output step 1:", output.shape)
        # reshape back to original (S, B, E) -> (B, S, E)

        if self.hparams.cls:
            output = output[0]
            print(output.shape)
        output = rearrange(output, 's b e -> b s e')
        output = rearrange(output, 'b s e -> b (s e)')
        print(output.shape)
        # output = torch.mean(output, dim=1)
        output = self.classifier(output)
        #output = torch.sigmoid(output)
        #output = F.softmax(output, dim=-1)
        # print(output[0])
        #output = torch.argmax(output, dim=-1)
        # print(output[0])
        return output

    def format_target(self, target):
        target = torch.cat(target, dim=0)
        target = target.squeeze()
        return target

