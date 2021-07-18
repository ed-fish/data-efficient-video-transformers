import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import device, Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer
import torchmetrics
#from torchmetrics.functional import accuracy
from pytorch_lightning.metrics.functional import accuracy
from models.contrastivemodel import OnlineEval


class SSLOnlineEval(Callback):
    """
    Attached mlp for fine tuning - edited version from ligtning sslonlineval docs
    """

    def __init__(self, drop_p = 0.2, z_dim = None, num_classes = None, model="MIT"):
        super().__init__()

        self.drop_p = drop_p
        self.optimizer: Optimizer

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss()
        self.f1 = torchmetrics.F1(num_classes=22)
        self.acc = torchmetrics.Accuracy()

    def on_pretrain_routine_start(self, trainer, pl_module):
        pl_module.non_linear_evaluator = OnlineEval(n_input=self.z_dim, n_classes=self.num_classes,p=self.drop_p).to(pl_module.device)
        self.optimizer = torch.optim.SGD(pl_module.non_linear_evaluator.parameters(), lr=1e-3)

    def get_representations(self, pl_module, x):
        representations, _ = pl_module(x)
        return representations


    def to_device(self, batch, device):

        x_i_experts = batch["x_i_experts"]
        label = batch["label"]

        x_i_experts = [torch.cat(x, dim=-1) for x in x_i_experts]
        x_i_input = torch.stack(x_i_experts)
        labels = torch.stack(label)

        x_i_input = x_i_input.to(device)
        labels = labels.to(device)

        return x_i_input, labels


    def on_train_batch_end(
            self, 
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            data_loader_idx
            ):

        x, labels = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()


        logits = pl_module.non_linear_evaluator(representations)
        mlp_loss = self.loss(logits, labels)
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        logits = logits.cpu()
        labels = labels.cpu()

        #accuracy = self.f1(logits, labels)
        acc = accuracy(logits, labels, num_classes=305)

        #pl_module.log("online f1 train", accuracy, on_step=True, on_epoch=False, sync_dist=True)
        pl_module.log("online train acc", acc, step=trainer.globaal_step)

    def on_validation_batch_end(
            self, 
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            data_loader_idx
            ):


        x, labels = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()


        logits = pl_module.non_linear_evaluator(representations)
        mlp_loss = self.loss(logits, labels)
        logits = logits.cpu()
        labels = labels.cpu()

        #accuracy = self.f1(logits, labels)
        acc = accuracy(logits, labels, num_classes=305)

        #pl_module.log("online f1 train", accuracy, on_step=True, on_epoch=False, sync_dist=True)
        pl_module.log("online val acc", acc, step=trainer.globaal_step)


        
