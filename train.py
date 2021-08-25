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

class CollaborativeGating(pl.LightningModule):
    def __init__(self):

        super(CollaborativeGating, self).__init__()
        self.proj_input = 2048
        self.proj_embedding_size = 2048
        self.projection = nn.Linear(self.proj_input, self.proj_embedding_size)
        self.cg = ContextGating(self.proj_input)
        self.geu = GatedEmbeddingUnit(self.proj_input, 1024,  False)

    def pad(self, tensor):

#        pad = (2048 - tensor.shape[1]) / 2
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

# take a list of experts [y1, y2, y3, y4]
# aggregate to a common dimension [128?, 512?, 1024?]
# for each expert
    # y1 -> [1, 1024]
    # y2 -> [1, 1024]
    # t1 = [y1 + y2] -> g0 -> [512]
    # y1 + (try concat and adding) y3 -> t2
    # y1 + (try concat and adding) y4 -> t3
    # theta = t1 + t2 + t3
    # y1^ = y1 hadamard sigmoid(theta)
    # y1^, y2^, y3^, y4^ -> GEM -> L2 norm expert
    # -> y of shape [1, 2048]


class TransformerModel(pl.LightningModule):

    def __init__(self, ntoken, ninp, nhead=4, nhid=2048, nlayers=4,batch_size=32, learning_rate=0.05, dropout=0.5, warmup_epochs=10, max_epochs=100, seq_len=5, momentum = 0, weight_decay=0, scheduling=False, token_embedding=15, architecture=None, mixing=None):
        super(TransformerModel, self).__init__()

        self.save_hyperparameters('nhead','nhid',"ninp", "ntoken",'learning_rate','batch_size','dropout', 'warmup_epochs', 'max_epochs', 'momentum','weight_decay', 'scheduling', 'token_embedding', "architecture")

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.seq_len = seq_len
        self.best_auc = 0

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=seq_len) # shared dropout value for pe and tm(el)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp).to(self.device)
        self.self_sup_encoder = SpatioTemporalContrastiveModel()
        #self.accuracy = torchmetrics.F1(num_classes=15, threshold=0.1, top_k=3) # f1 weighted for mmx
        #self.accuracy = torchmetrics.Accuracy()
        self.decoder = nn.Linear(ninp, token_embedding)
        self.classifier = nn.Linear(token_embedding * seq_len, ntoken)
        self.mixing = mixing
        self.architecture = architecture
        self.collab = CollaborativeGating()
        self.bs = batch_size
        self.init_weights()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduling == True:
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
        else:
            return optimizer

    def contrastive_forward(self, data):
        encoded_list = []
        for d in data:
            e, d = self.self_sup_encoder(d)
            encoded_list.append(e)
        return encoded_list

    def shared_step(self, data, target):

        if self.pre_contrastive:
            data = self.contrastive_forward(data)

        if self.mixing== "collab" and self.architecture=="pre-trans":
            data = self.collab(data)
        else:
            data = torch.cat(data, dim=0)

        target = torch.cat(target, dim=0)

        # Permute to [ S, B, E ]
        print(data.shape)
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

        transform_t = transform_t.reshape(self.bs, -1)
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

        loss = self.criterion(data, target)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        #acc_preds = self.preds_acc(data)
        self.log('train/f1@t1', f1(data, target.to(int), num_classes=15, threshold=-1.0), on_step=False, on_epoch=True)
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
        self.log('val/f1@t1', f1(data, target.to(int), num_classes=15, average="weighted",threshold=-1.0), on_step=False, on_epoch=True)
        self.log('val/f1@t-1.5', f1(data, target.to(int), num_classes=15,average="weighted", threshold=-1.5), on_step=False, on_epoch=True)
        self.log('val/f1@t-2.0', f1(data, target.to(int), num_classes=15,average="weighted", threshold=-2.0),on_step=False, on_epoch=True)
        self.log('val/f1@t0', f1(data, target.to(int), num_classes=15,average="weighted", threshold=0),on_step=False, on_epoch=True)
        #print(target.shape)
        #print(target.to(int))
        # area_under_curve = auroc(F.softmax(data), target.to(int),num_classes=15, average="weighted")
        # self.log('val/auprc', area_under_curve,on_step=False, on_epoch=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
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
    wandb_logger = WandbLogger(project="MMX_Temporal", log_model='all')

    # dm = MITDataModule("data/mit/mit_tensors_train_wc.pkl","data/mit/mit_tensors_train_wc.pkl", config)
    # dm = MMXDataModule("data/mmx/mmx_tensors_val.pkl","data/mmx/mmx_tensors_val.pkl", config)

    bptt = config["batch_size"].get()
    learning_rate = config["learning_rate"].get()
    scheduling = config["scheduling"].get()
    momentum = config["momentum"].get()
    weight_decay = config["weight_decay"].get()
    token_embedding = config["token_embedding"].get()
    experts = config["experts"].get()
    epochs = config["epochs"].get()
    n_warm_up = 70
    seq_len = config["seq_len"].get()
    ntokens = config["n_labels"].get()
    emsize = config["input_shape"].get()
    mixing_method = config["mixing_method"].get()
    nhid = 1850
    nlayers = 3
    frame_agg = config["frame_agg"].get()
    nhead = config["n_heads"].get()
    dropout = config["dropout"].get()
    frame_id = config["frame_id"].get()
    cat_norm = config["cat_norm"].get()
    cat_softmax = config["cat_softmax"].get()
    architecture = config["architecture"].get()

    params = { "experts":experts,
               "input_shape": config["input_shape"].get(),
               "mixing_method":mixing_method,
               "epochs": epochs,
               "frame_id":frame_id,
               "batch_size": bptt,
               "seq_len": seq_len,
               "nlayers":nlayers,
               "dropout":dropout,
               "cat_norm": cat_norm,
               "cat_softmax": cat_softmax,
               "nhid":nhid,
               "nhead":nhead,
               "n_warm_up":n_warm_up,
               "learning_rate":learning_rate,
               "scheduling":scheduling,
               "weight_decay":weight_decay, 
               "momentum":momentum,
               "token_embedding":token_embedding,
               "architecture":architecture,
               "frame_agg":frame_agg }

    wandb.init(config=params)
    config = wandb.config
    # dm = MMXDataModule("data_processing/trailer_temporal/mmx_tensors_train.pkl", "data_processing/trailer_temporal/mmx_tensors_val.pkl", config)
    dm = MMXDataModule("data_processing/trailer_temporal/train_tst.pkl", "data_processing/trailer_temporal/val_tst.pkl", config)

    model = TransformerModel(ntokens, emsize, config["nhead"], nhid=config["nhid"],batch_size=config["batch_size"], nlayers=config["nlayers"], learning_rate=config["learning_rate"], 
                             dropout=config["dropout"], warmup_epochs=config["n_warm_up"], max_epochs=config["epochs"], seq_len=config["seq_len"], token_embedding=config["token_embedding"], architecture=config["architecture"],mixing = config["mixing_method"])
    trainer = pl.Trainer(gpus=[3], callbacks=[lr_monitor], max_epochs=epochs, logger=wandb_logger)
    trainer.fit(model, datamodule=dm)

train()
