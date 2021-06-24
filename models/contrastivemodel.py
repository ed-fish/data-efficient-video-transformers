import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import math
# from models.pretrained.models import EmbeddingExtractor
from models.losses.ntxent import ContrastiveLoss

class SpatioTemporalContrastiveModel(pl.LightningModule):
    def __init__(self, config):
        super(SpatioTemporalContrastiveModel, self).__init__()
        self.input_layer_size = config["input_shape"].get()
        self.bottleneck_size = config["bottle_neck"].get()
        self.output_layer_size = config["output_shape"].get()
        self.batch_size = config["batch_size"].get()
        self.config = config
        # self.ee = EmbeddingExtractor(self.config)

        self.fc1 = nn.Linear(self.input_layer_size, self.input_layer_size)
        self.fc2 = nn.Linear(self.input_layer_size, self.bottleneck_size)
        self.fc3 = nn.Linear(self.bottleneck_size, self.bottleneck_size)
        self.fc4 = nn.Linear(self.bottleneck_size, self.output_layer_size)
        self.loss = ContrastiveLoss(self.batch_size)
        self.proj_list = []

    def forward(self, tensor):
        output = F.relu(self.fc1(tensor))
        output = F.relu(self.fc2(output))
        embedding = F.relu(self.fc3(output))
        output = self.fc4(output)

        return embedding, output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config["learning_rate"].get())
        return optimizer

    def expert_aggregation(self, expert_list):
        agg = self.config["aggregation"].get()

        if agg == "avg_pool":
            expert_list = torch.cat(expert_list, dim=-1)
            expert_list = F.adaptive_avg_pool2d(expert_list, self.input_layer_size)

        if agg == "mean_pool":
            expert_list = torch.cat(expert_list, dim=-1)
            expert_list = F.adaptive_max_pool2d(expert_list, size)

        if agg == "concat":
            expert_list = torch.cat(expert_list, dim=-1)

        return expert_list

    def debug(self, x_i, x_j):
        for keys, values in x_i.items():
            print(keys, values.shape)

    def training_step(self, batch, batch_idx):
        x_i_experts, x_j_experts = batch
        #x_i_input = self.expert_aggregation(x_i_experts).squeeze(1)
        #x_j_input = self.expert_aggregation(x_j_experts).squeeze(1)

        x_i_embedding, x_i_out = self(x_i_experts)
        x_j_embedding, x_j_out = self(x_j_experts)

        x_i_out = x_i_out.squeeze()
        x_j_out = x_j_out.squeeze()

        loss = self.loss(x_i_out, x_j_out)
        self.log("training loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x_i_experts, _ = batch
        x_i_embeddings, _ = self(x_i_experts)
        x_i_out = x_i_embeddings.squeeze()
        self.proj_list.append(x_i_out)    
        list_len = len(self.proj_list)
        return {"length": list_len}


