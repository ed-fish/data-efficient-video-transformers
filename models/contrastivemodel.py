import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lars import LARS
# from models.pretrained.models import EmbeddingExtractor
from models.losses.ntxent import ContrastiveLoss

class SpatioTemporalContrastiveModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.input_layer_size = 2048
        self.hidden_layer_size = 1024
        self.projection_size = 512
        self.output_layer_size = 128
        self.batch_size = config["batch_size"]
        self.num_samples = config["num_samples"]
        self.config = config
        self.train_iters_per_epoch = self.num_samples // self.batch_size
        self.running_logits = []
        self.running_labels = []
        # self.ee = EmbeddingExtractor(self.config)

        self.encoder_net = nn.Sequential(
                nn.Linear(self.input_layer_size, self.hidden_layer_size, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(self.hidden_layer_size),
                nn.Linear(self.hidden_layer_size, self.hidden_layer_size, bias=False),
                nn.ReLU(inplace=True),
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
       # embedding = F.normalize(embedding)
       output = self.projector_net(embedding)

       return embedding, output

    def configure_optimizers(self):
        # parameters = self.exclude_from_wt_decay(
        #     self.named_parameters(),
        #     weight_decay=self.config["weight_decay"]
        # )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        # optimizer = LARS(
        #     parameters,
        #     lr=self.config["learning_rate"],
        #     momentum=self.config["momentum"],
        #     weight_decay=self.config["weight_decay"],
        #     trust_coefficient=0.0001,
        #     )

        # Trick 2 (after each step)
        # self.hparams.warmup_epochs = self.config["warm_up"] * self.train_iters_per_epoch
        # max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        # linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=2,
        #     max_epochs=10,
        #     warmup_start_lr=0,
        #     eta_min=0
        # )

        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.config["epochs"] // 10, max_epochs=self.config["epochs"])

        # scheduler = {
        #     'scheduler': linear_warmup_cosine_decay,
        #     'interval': 'step',
        #     'frequency': 1
        # }

        return [optimizer], [scheduler]
        # optimizer = torch.optim.Adam(self.parameters(),
        #                              lr=self.config["learning_rate"])
	
        # return optimizer


    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]

    def expert_aggregation(self, expert_list):
        agg = self.config["aggregation"]

        # in the case that there is just one expert 
        if agg == "none":
            expert_list = expert_list[0]

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

        x_i_out = F.normalize(x_i_out.squeeze())
        x_j_out = F.normalize(x_j_out.squeeze())

        loss = self.loss(x_i_out, x_j_out)
        self.log("train/contrastive/loss", loss)
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

        self.log("val/contrastive/loss", loss)
        return {"loss":loss, "val_outputs":x_i_embedding}

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


    # def validation_epoch_end(self, val_step_outputs):
    #     print(val_step_outputs[0]["val_outputs"].shape)


