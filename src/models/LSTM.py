import torch
import torch.nn as nn
import pytorch_lightning as pl


class LSTMRegressor(pl.LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''

    def __init__(self,
                 n_features,
                 hidden_size,
                 seq_len,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 criterion):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.running_logits = []
        self.running_labels = []
        self.learning_rate = learning_rate
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=num_layers,
                            dropout=dropout)
        self.linear = nn.Linear(hidden_size, 15)

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        # print(x.shape)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x = batch["experts"]
        x = torch.stack(x).squeeze(1)
        y = batch["label"]
        y = torch.cat(y).squeeze(1)
        y = y.float()
        y_hat = self(x)
        y_hat = torch.sigmoid(y_hat)
        loss = self.criterion(y_hat, y)
        #result = pl.TrainResult(loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["experts"]
        x = torch.stack(x, dim=0).squeeze(1)
        y = batch["label"]
        y = torch.cat(y).squeeze(1)
        y = y.float()
        y_hat = self(x)
        y_hat = torch.sigmoid(y_hat)
        loss = self.criterion(y_hat, y)
        print("-" * 20)
        print(y[0])
        print(y_hat[0])
        print("-" * 20)
        self.running_labels.append(y)
        self.running_logits.append(y_hat)
        #result = pl.EvalResult(checkpoint_on=loss)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.running_labels.append(y)
        self.running_logits.append(y_hat)
        result.log('test_loss', loss)
        return result
