import confuse 
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb
from torchmetrics.functional import f1, auroc
from pytorch_lightning.loggers import WandbLogger
from dataloaders.MIT_Temporal_dl import MITDataset, MITDataModule
from models.contrastivemodel import SpatioTemporalContrastiveModel
from dataloaders.MMX_Temporal_dl import MMXDataset, MMXDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from callbacks.callbacks import TransformerEval

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
        self.mean = 0.06
        self.std = 0.2

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CollaborativeGating(pl.LightningModule):
    def __init__(self):
        super(CollaborativeGating, self).__init__()
        self.proj_input = 2048
        self.proj_embedding_size = 2048
        self.projection = nn.Linear(self.proj_input, self.proj_embedding_size)
        self.cg = ContextGating(self.proj_input)
        self.geu = GatedEmbeddingUnit(self.proj_input, 1024,  False)

    def pad(self, tensor):
        tensor = tensor.unsqueeze(0)
        curr_expert = F.interpolate(tensor, 2048)
        curr_expert = curr_expert.squeeze(0)
        return curr_expert

    def forward(self, batch):
        batch_list = []
        for scenes in batch: # this will be batches
            scene_list = []
            # first expert popped off
            for experts in scenes:
                expert_attention_vec = []
                for i in range(len(experts)):
                    curr_expert = experts.pop(0)
                    if curr_expert.shape[1] != 2048:
                        curr_expert = self.pad(curr_expert)

                    # compare with all other experts
                    curr_expert = self.projection(curr_expert)
                    t_i_list = []
                    for c_expert in experts:
                        # through g0 to get feature embedding t_i
                        if c_expert.shape[1] != 2048:
                            c_expert = self.pad(c_expert)
                        c_expert = self.projection(c_expert)
                        t_i = curr_expert + c_expert # t_i maps y1 to y2
                        t_i_list.append(t_i)
                    t_i_summed = torch.stack(t_i_list, dim=0).sum(dim=0) # all other features
                    expert_attention = self.projection(t_i_summed) # attention vector for all comparrisons
                    expert_attention_comp = self.cg(curr_expert, expert_attention) # gated version
                    expert_attention_vec.append(expert_attention_comp)
                    experts.append(curr_expert)
                expert_attention_vec = torch.stack(expert_attention_vec, dim=0).sum(dim=0) # concat all attention vectors
                expert_vector = self.geu(expert_attention_vec) # apply gated embedding
                scene_list.append(expert_vector)
                expert_vector = self.geu(expert_attention_vec) # apply gated embedding
                scene_list.append(expert_vector)
            scene_stack = torch.stack(scene_list)
            batch_list.append(scene_stack)
        batch = torch.stack(batch_list, dim = 0)
        batch = batch.squeeze(2)
        return batch


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension, use_bn):
        super(GatedEmbeddingUnit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        # self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)

    def forward(self, x):
        x = self.fc(x)
        #x = self.cg(x)
        x = F.normalize(x)
        return x

class ContextGating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGating, self).__init__()
        # self.add_batch_norm = add_batch_norm
        # self.batch_norm = nn.BatchNorm1d(dimension)
        # self.batch_norm2 = nn.BatchNorm1d(dimension)

    def forward(self, x, x1):

        # if self.add_batch_norm:
        #     x = self.batch_norm(x)
        #     x1 = self.batch_norm2(x1)
        t = x + x1
        x = torch.cat((x, t), -1)
        return F.glu(x, -1)


class TransformerModel(pl.LightningModule):

    def __init__(self, 
                 ntoken, 
                 ninp, 
                 nhead=4, 
                 nhid=2048,
                 nlayers=4,
                 batch_size=32, 
                 learning_rate=0.05, 
                 dropout=0.5,
                 warmup_epochs=10, 
                 max_epochs=100, 
                 seq_len=5, 
                 momentum = 0,
                 weight_decay=0, 
                 scheduling=False, 
                 token_embedding=15,
                 architecture=None, 
                 mixing=None):
        super(TransformerModel, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=seq_len) # shared dropout value for pe and tm(el)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp).to(self.device)
        self.decoder = nn.Linear(ninp, token_embedding)
        self.classifier = nn.Sequential(
                nn.Linear(4024, 1024, bias=False),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(128, ntoken))
        self.mixing = mixing
        self.architecture = architecture
        self.collab = CollaborativeGating()
        self.bs = batch_size
        self.init_weights()
        self.running_labels = []
        self.running_logits = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def shared_step(self, data, target):
        # flatten or mix output embeddings
        if self.mixing== "collab" and self.architecture=="pre-trans":
            data = self.collab(data)
        else:
            data = torch.cat(data, dim=0)
        ##print("shared step 1:", data.shape)

        target = torch.cat(target, dim=0)

        # reshape for transformer output (B, S, E) -> (S, B, E)
        data = data.permute(1, 0, 2)
        src_mask = self.generate_square_subsequent_mask(data.size(0))
        src_mask = src_mask.to(self.device)

        # FORWARD
        output = self(data, src_mask)

        ##print("output step 1:", output.shape)

        # reshape back to original (S, B, E) -> (B, S, E)
        transform_t = output.permute(1, 0, 2)

        #print("output_reshape", output.shape)

        # flatten sequence embeddings (S, B, E) -> (B, S * E)
        transform_t = transform_t.reshape(self.bs, -1)

        #print("output_reshape 2", output.shape)
        transform_t = transform_t.unsqueeze(0)

        # Pooling before classification?
        transform_t = F.adaptive_avg_pool1d(transform_t, 4024)

        #print("output pool", transform_t.shape)
        transform_t = transform_t.squeeze(0)

        # Send total embeddings to classifier - alternative to BERT Token
        pooled_result = self.classifier(transform_t)

        #print("output classifier", pooled_result.shape)

        # pooled_result = F.adaptive_avg_pool3d(transform_t, (32, 1, 15))
        # pooled_result = pooled_result.squeeze().squeeze()

        # Cross Entropy includes softmax https://bit.ly/3f73RJ7 - add here for others. 
        # reshape after pooling
        target = target.squeeze()

        return pooled_result, target

    def training_step(self, batch, batch_idx):
        data = batch["experts"]
        target = batch["label"]

        data, target = self.shared_step(data, target)
        target = target.float()

        loss = self.criterion(data, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        #acc_preds = self.preds_acc(data)

        # gradient clipping for stability
        # torch.nn.utils.clip_grad_norm(self.parameters(), 0.5)

        return loss

    def validation_step(self, batch, batch_idx):

        data = batch["experts"]
        target = batch["label"]

        data, target = self.shared_step(data, target)
        target = target.float()

        #target = torch.argmax(target, dim=-1)
        loss = self.criterion(data, target)
        target = F.softmax(target)
        
        # acc_preds = self.preds_acc(data)
        self.running_labels.append(target)
        self.running_logits.append(data)

        self.log("val/loss", loss, on_step=True, on_epoch=True)
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
        #src = self.encoder(src) * math.sqrt(self.hparams.ninp)
        src_mask = self.generate_square_subsequent_mask(self.seq_len)
        src = self.pos_encoder(src)
        src_mask = src_mask.to(self.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        # output = F.softmax(output)
        # Do not include softmax if nn.crossentropy as softmax included via NLLoss
        return output
