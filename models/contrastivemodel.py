import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import math
# from models.pretrained.models import EmbeddingExtractor
from models.losses.ntxent import ContrastiveLoss

class OnlineEval(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden

        self.block_forward = nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(n_hidden),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
                )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class SpatioTemporalContrastiveModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.input_layer_size = 2048
        self.hidden_layer_size = 1024
        self.projection_size = 512
        self.output_layer_size = 128
        self.batch_size = config["batch_size"]
        self.config = config
        # self.ee = EmbeddingExtractor(self.config)

        self.encoder_net = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.input_layer_size, self.hidden_layer_size, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(self.hidden_layer_size),
                nn.Dropout(p=0.1),
                nn.Linear(self.hidden_layer_size, self.projection_size),
                )

        self.projector_net = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(self.projection_size, self.projection_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(self.projection_size, self.output_layer_size),
                )

        self.loss = ContrastiveLoss(self.batch_size)

        self.proj_list = []
        self.label_list = []

    def forward(self, tensor):
       # if self.config["aggregation"].get() == "collab":
       embedding = self.encoder_net(tensor)
       output = self.projector_net(embedding)

       return embedding, output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config["learning_rate"])
        return optimizer

    def expert_aggregation(self, expert_list):
        agg = self.config["aggregation"]

        # in the case that there is just one expert 
        if agg == "none":
            pass

        if agg == "avg_pool":
            expert_list = torch.cat(expert_list, dim=-1)
            expert_list = F.adaptive_avg_pool2d(expert_list, self.input_layer_size)

        if agg == "mean_pool":
            expert_list = torch.cat(expert_list, dim=-1)
            expert_list = F.adaptive_max_pool2d(expert_list, size)

        if agg == "concat":
            expert_list = torch.cat(expert_list, dim=-1)

        if agg == "collab_gate":
            pass

        return expert_list

    def debug(self, x_i, x_j):
        for keys, values in x_i.items():
            print(keys, values.shape)

    def training_step(self, batch, batch_idx):
        x_i_experts = batch["x_i_experts"]
        x_j_experts = batch["x_j_experts"]
        label = batch["label"]

        #x_i_input = self.expert_aggregation(x_i_experts).squeeze(1)
        #x_j_input = self.expert_aggregation(x_j_experts).squeeze(1)

        x_i_experts = [self.expert_aggregation(x) for x in x_i_experts]
        x_j_experts = [self.expert_aggregation(x) for x in x_j_experts]

        x_i_experts = torch.stack(x_i_experts)
        x_j_experts = torch.stack(x_j_experts)


        x_i_experts = x_i_experts.squeeze()
        x_j_experts = x_j_experts.squeeze()

        x_i_embedding, x_i_out = self(x_i_experts)
        x_j_embedding, x_j_out = self(x_j_experts)

        x_i_out = x_i_out.squeeze()
        x_j_out = x_j_out.squeeze()

        loss = self.loss(x_i_out, x_j_out)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        x_i_experts = batch["x_i_experts"]
        x_j_experts = batch["x_j_experts"]
        label = batch["label"]

        #x_i_input = self.expert_aggregation(x_i_experts).squeeze(1)
        #x_j_input = self.expert_aggregation(x_j_experts).squeeze(1)

        x_i_experts = [self.expert_aggregation(x) for x in x_i_experts]
        x_j_experts = [self.expert_aggregation(x) for x in x_j_experts]

        x_i_experts = torch.stack(x_i_experts)
        x_j_experts = torch.stack(x_j_experts)

        x_i_experts = x_i_experts.squeeze()
        x_j_experts = x_j_experts.squeeze()

        x_i_embedding, x_i_out = self(x_i_experts)
        x_j_embedding, x_j_out = self(x_j_experts)

        x_i_out = x_i_out.squeeze()
        x_j_out = x_j_out.squeeze()

        loss = self.loss(x_i_out, x_j_out)
        self.log("training loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x_i_experts = batch["x_i_experts"]

        x_i_experts = [self.expert_aggregation(x) for x in x_i_experts]
        x_i_experts = torch.stack(x_i_experts)
        label = batch["label"]
        x_i_embeddings, _ = self(x_i_experts)
        x_i_out = x_i_embeddings.squeeze()
        self.proj_list.append(x_i_out)    
        self.label_list.append(label)
        list_len = len(self.proj_list)
        return {"length": list_len}


