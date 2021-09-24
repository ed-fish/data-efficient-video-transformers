import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import device, Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional import auroc
from torchmetrics import ConfusionMatrix
import pandas as pd
import torchmetrics
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, recall_score, average_precision_score, precision_score
from torchmetrics.functional import f1, auroc
#from torchmetrics.functional import accuracy
import torchmetrics

class TransformerEval(Callback):
    def __init__(self):
        self.best_acc = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        running_labels = torch.tensor(pl_module.running_labels)
        running_logits = torch.tensor(pl_module.running_logits)
        acc = torch.sum(running_logits==running_labels).item() / (len(running_labels) * 1.0)
        pl_module.log("val/accuracy/epoch", acc, on_step=False, on_epoch=True) 
        print(f"acc:{acc} len_S:{len(running_labels)} ex:{running_labels[0]} : {running_logits[0]}")
        pl_module.running_labels = []
        pl_module.running_logits = []

        if acc > self.best_acc:
            trainer.save_checkpoint("mit_location_acc.ckkpt")
            self.best_acc = acc

        


class SSLOnlineEval(Callback):
    """
    Attached mlp for fine tuning - edited version from ligtning sslonlineval docs
    """

    def __init__(self, drop_p = 0.1, z_dim = None, num_classes = None, model="MIT"):
        super().__init__()

        self.drop_p = drop_p
        self.optimizer: Optimizer
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=305)

    def on_pretrain_routine_start(self, trainer, pl_module):
        from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
        pl_module.non_linear_evaluator = SSLEvaluator(n_input=self.z_dim, n_classes=self.num_classes,p=self.drop_p).to(pl_module.device)
        pl_module.non_linear_evaluator.to(pl_module.device)
        self.optimizer = torch.optim.SGD(pl_module.non_linear_evaluator.parameters(), lr=0.005)
        self.accuracy = self.accuracy.to(pl_module.device)

    def get_representations(self, pl_module, x):
        x = x.squeeze()
        representations, _ = pl_module(x)
        return representations

    def to_device(self, batch, device):

        x_i_experts = batch["x_i_experts"]
        labels = batch["label"]
        x_i_experts = [x[0] for x in x_i_experts]
        x_i_experts = torch.stack(x_i_experts)
        x_i_experts = x_i_experts.squeeze()

        labels = [x[0] for x in labels]
        labels = torch.stack(labels)
        labels = labels.squeeze()

        x_i_input = x_i_experts.to(device)
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
        # pl_module.running_logits.append(logits)
        # pl_module.running_labels.append(labels)
        mlp_loss = self.loss(logits, labels)
        pl_module.log("train/online/loss", mlp_loss)
        mlp_loss.backward()
        self.optimizer.step()


    def on_validation_epoch_end(self, trainer, pl_module):
        total_acc = self.accuracy.compute()
        pl_module.log("val/online/epochacc", total_acc, on_epoch=True)
        self.accuracy.reset()

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
        pl_module.log("val/online/loss", mlp_loss)
        logits = F.softmax(logits)
        accuracy = self.accuracy(logits, labels) 
        pl_module.log("val/online/accuracy", accuracy)
