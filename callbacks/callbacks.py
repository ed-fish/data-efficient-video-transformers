import torch
from torch import device, Tensor
import pickle
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional import auroc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
# from torchmetrics import ConfusionMatrix
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
#from pytorch_lightning.metrics.functional import accuracy


class TransformerEval(Callback):

    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_shared_end(pl_module, "val")

    def on_shared_end(self, pl_module, state):

        target_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                        'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Science Fiction', 'Thriller',  'War']
        running_labels = torch.cat(pl_module.running_labels)
        running_logits = torch.cat(pl_module.running_logits)
        t = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        running_logits = running_logits.cpu()
        running_labels = running_labels.cpu()
        for threshold in t:
            accuracy = f1_score(running_labels.to(int), (running_logits > threshold).to(
                int), average="weighted", zero_division=0)
            # recall = recall_score(running_labels.to(int), (running_logits > threshold).to(int), average="weighted", zero_division=1)
            # precision = precision_score(running_labels.to(int), (running_logits > threshold).to(int), average="weighted", zero_division=1)
            # avg_precision = average_precision_score(running_labels.to(int), (running_logits > threshold).to(int), average="weighted")

            pl_module.log(f"{state}/online/f1@{str(threshold)}", accuracy)
            #pl_module.log(f"{state}/online/recall@{str(threshold)}", recall, on_epoch=True)
            #pl_module.log(f"{state}/online/precision@{str(threshold)}", precision, on_epoch=True)
            #pl_module.log(f"{state}/online/avg_precision@{str(threshold)}", avg_precision, on_epoch=True)

        pl_module.running_labels = []
        pl_module.running_logits = []

        label_str = []
        target_str = []

    def translate_labels(self, label_vec):
        target_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                        'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Science Fiction', 'Thriller',  'War']
        labels = []
        for i, l in enumerate(label_vec):
            if l:
                labels.append(target_names[i])
        return labels


class MITEval(Callback):
    def __init__(self):
        self.best_acc = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        running_labels = torch.cat(pl_module.running_labels)
        running_logits = torch.cat(pl_module.running_logits)
        acc = torch.sum(running_logits == running_labels).item() / \
            (len(running_labels) * 1.0)
        pl_module.log("val/accuracy/epoch", acc, on_step=False, on_epoch=True)
        print(
            f"acc:{acc} len_S:{len(running_labels)} ex:{running_labels[0]} : {running_logits[0]}")
        pl_module.running_labels = []
        pl_module.running_logits = []

        # if acc > self.best_acc:
        #     trainer.save_checkpoint("mit_location_acc.ckkpt")
        #     self.best_acc = acc


class DisplayResults(Callback):
    def on_test_end(self, trainer, pl_module):

        cache_dict = {}

        running_logits = pl_module.running_logits
        running_labels = pl_module.running_labels
        running_embeds = pl_module.running_embeds
        running_paths = pl_module.running_paths

        # 52 10 15

        running_labels = torch.cat(running_labels).cpu().numpy()
        running_embeds = torch.cat(running_embeds).cpu().numpy()
        running_logits = torch.cat(running_logits).cpu().numpy()
        running_paths = [x for i in running_paths for x in i]
        running_logits = np.where(running_logits > 0.3, 1, 0)
        running_labels = running_labels.astype(int)

        for x, i in enumerate(running_labels):
            print("paths", x, running_paths[x])
            print("actual", x, ":", self.n_to_labels(i))
            print("predicted", x, ":", self.n_to_labels(running_logits[x]))
            print("embedding", x, ":", running_embeds[x])
            cache_dict[x] = {"path": running_paths[x], "embedding": running_embeds[x],
                             "predicted": self.n_to_labels(running_logits[x]), "actual": self.n_to_labels(i)}
        with open("embed_dict", "wb") as file:
            pickle.dump(cache_dict, file)

            # print(running_logits)

    def n_to_labels(self, vector):
        labels = []

        target_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                        'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Science Fiction', 'Thriller',  'War']
        for i, x in enumerate(vector):
            if x:
                labels.append(target_names[i])
        return labels


class SSLOnlineEval(Callback):
    """
    Attached mlp for fine tuning - edited version from ligtning sslonlineval docs
    """

    def __init__(self, drop_p=0.1, z_dim=None, num_classes=None, model="MIT"):
        super().__init__()

        self.drop_p = drop_p
        self.optimizer: Optimizer

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.loss = nn.BCELoss()

    def on_pretrain_routine_start(self, trainer, pl_module):
        from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=self.z_dim, n_classes=self.num_classes, p=self.drop_p).to(pl_module.device)
        self.optimizer = torch.optim.SGD(
            pl_module.non_linear_evaluator.parameters(), lr=0.005)

    def get_representations(self, pl_module, x):
        x = x.squeeze()
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
        logits = torch.sigmoid(logits)

        # pl_module.running_logits.append(logits)
        # pl_module.running_labels.append(labels)

        mlp_loss = self.loss(logits, labels)
        pl_module.log("train/online/loss", mlp_loss)
        mlp_loss.backward()
        self.optimizer.step()

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
        logits = torch.sigmoid(logits)

        mlp_loss = self.loss(logits, labels)
        pl_module.log("val/online/loss", mlp_loss)
        logits = logits.cpu()
        labels = labels.cpu()

        pl_module.running_logits.append(logits)
        pl_module.running_labels.append(labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_shared_end(pl_module, "val")

    # def on_train_epoch_end(self, trainer, pl_module):
    #     self.on_shared_end(pl_module, "train")

    def on_shared_end(self, pl_module, state):

        target_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                        'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Science Fiction', 'Thriller',  'War']
        running_labels = torch.cat(pl_module.running_labels)
        running_logits = torch.cat(pl_module.running_logits)
        # running_logits = F.sigmoid(running_logits)
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        for t in thresholds:
            accuracy = f1_score(running_labels.to(int), (running_logits > t).to(
                int), average="weighted", zero_division=1)
            recall = recall_score(running_labels.to(int), (running_logits > t).to(
                int), average="weighted", zero_division=1)
            precision = precision_score(running_labels.to(
                int), (running_logits > t).to(int), average="weighted", zero_division=1)
            avg_precision = average_precision_score(running_labels.to(
                int), (running_logits > t).to(int), average="weighted")

            pl_module.log(f"{state}/online/f1@{str(t)}",
                          accuracy, on_epoch=True)
            pl_module.log(f"{state}/online/recall@{str(t)}",
                          recall, on_epoch=True)
            pl_module.log(f"{state}/online/precision@{str(t)}",
                          precision, on_epoch=True)
            pl_module.log(f"{state}/online/avg_precision@{str(t)}",
                          avg_precision, on_epoch=True)

        running_labels = running_labels.to(int).numpy()
        running_logits = (running_logits > 0.1).to(int).numpy()

        pl_module.running_labels = []
        pl_module.running_logits = []

        label_str = []
        target_str = []

        test_table = wandb.Table(columns=["truth", "guess"])

        for i in range(0, 20):
            test_table.add_data(self.translate_labels(
                running_labels[i]), self.translate_labels(running_logits[i]))

        pl_module.logger.experiment.log({"table": test_table})

    def translate_labels(self, label_vec):
        target_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                        'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Science Fiction', 'Thriller',  'War']
        labels = []
        for i, l in enumerate(label_vec):
            if l:
                labels.append(target_names[i])
        return labels
