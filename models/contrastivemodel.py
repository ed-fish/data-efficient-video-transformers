import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SpatioTemporalContrastiveModel(pl.LightningModule):
    def __init__(self, config):
        super(SpatioTemporalContrastiveModel, self).__init__()
        self.input_layer_size = config["input_shape"].get(
        ) * len(config["experts"].get())
        self.bottleneck_size = config["bottle_neck"].get()
        self.output_layer_size = config["output_shape"].get()
        self.batch_size = config["batch_size"].get()

        self.fc1 = nn.Linear(self.input_layer_size, self.input_layer_size)
        self.fc2 = nn.Linear(self.input_layer_size, self.bottleneck_size)
        self.fc3 = nn.Linear(self.bottleneck_size, self.bottleneck_size)
        self.fc4 = nn.Linear(self.bottleneck_size, self.output_layer_size)

    def forward(self, tensor):
        output = F.relu(self.fc1(tensor))
        output = F.relu(self.fc2(output))
        embedding = F.relu(self.fc3(output))
        output = self.fc4(output)

        return embedding, output

    def configure_optimizers(self):
        optimizer = NT_Xent(self.batch_size, 0.5, 1)
        return optimizer

    def training_step(self, batch, batch_idx):
        label = batch["label"]
        motion = batch["motion"]
        location = batch["location"]
        audio = batch["audio"]
        image = batch["image"]
