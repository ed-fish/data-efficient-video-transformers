import torch.nn as nn
import torchmetrics
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import math
# from models.pretrained.models import EmbeddingExtractor
from models.losses.ntxent import ContrastiveLoss

class BasicMLP(pl.LightningModule):
    def __init__(self, config):
        super(BasicMLP, self).__init__()
        self.input_layer_size = config["input_shape"].get()
        self.bottleneck_size = config["bottle_neck"].get()
        self.output_layer_size = config["output_shape"].get()
        self.batch_size = config["batch_size"].get()
        self.config = config
        # self.ee = EmbeddingExtractor(self.config)
        self.f1 = torchmetrics.F1(num_classes=21)

        self.fc1 = nn.Linear(self.input_layer_size, self.input_layer_size)
        self.fc2 = nn.Linear(self.input_layer_size, self.bottleneck_size)
        self.fc3 = nn.Linear(self.bottleneck_size, self.bottleneck_size)
        self.fc4 = nn.Linear(self.bottleneck_size, 21)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, tensor):
        output = F.relu(self.fc1(tensor))
        output = F.relu(self.fc2(output))
        embedding = F.relu(self.fc3(output))
        output = self.fc4(embedding)
        return output

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
        x_i_experts = batch["x_i_experts"]
        label = batch["label"]

        x_i_experts = [self.expert_aggregation(x) for x in x_i_experts]
        #x_j_input = self.expert_aggregation(x_j_experts).squeeze(1)

        x_i_input = torch.stack(x_i_experts)
        labels = torch.stack(label)

        output = self(x_i_input)
        output = output.squeeze()
        loss = self.loss(output, labels)
        self.log("training loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_i_experts = batch["x_i_experts"]
        label = batch["label"]

        x_i_experts = [self.expert_aggregation(x) for x in x_i_experts]
        #x_j_input = self.expert_aggregation(x_j_experts).squeeze(1)

        x_i_input = torch.stack(x_i_experts)
        labels = torch.stack(label)

        output = self(x_i_input)
        output = output.squeeze()
        loss = self.loss(output, labels)
        self.log("validation loss", loss)
        accuracy = self.f1(output, labels)
        self.log("f1 score", accuracy)
        print(accuracy)
        return loss
        


