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
        print(f"motion {motion.shape} \n location {location.shape} \n audio {audio.shape} \n image {image.shape}")


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(
            1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
