"""
Algorithm: linear_regression(线性回归)
数据集:波士顿房价数据集

编写过程:
1.编写model,确定输入输出
2.编写数据加载方式、dataSet、dataLoader
3.fit()
import pytorch_lightning as pl
"""
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.utils import shuffle as sk_shuffle
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader


def load_data():
    """
    创建数据集: y = 3x + 0.4 + noise
    """
    count = 30
    noise = np.random.randint(-50, 50, size=count) / 100
    x = np.linspace(1, 100, count)
    y = 3 * x + 2 + noise
    print(x)
    print(y)
    return x, y


def show():
    x, y = load_data()
    plt.plot(x, y, 'ro', label='Original data')
    # plt.plot(x_train.numpy(), predict, label='Fitting Line')
    # 显示图例
    plt.legend()
    plt.show()


class SklearnDataSet(Dataset):
    def __init__(self, X, Y, shuffle=False, random_state=1234):
        super().__init__()
        if shuffle:
            X, Y = sk_shuffle(X, Y, random_state=random_state)
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx]

        # Do not convert integer to float for classification data
        if not y.dtype == np.integer:
            y = y.astype(np.float32)
        return x, y


class LinearRegression(pl.LightningModule):

    def __init__(self,
                 input_dim: int,
                 bias: bool = True,
                 learning_rate: float = 0.0001,
                 optimizer: Optimizer = Adam,
                 l1_strength: float = None,
                 l2_strength: float = None,
                 **kwargs):
        """
                Linear regression model implementing - with optional L1/L2 regularization
                $$min_{W} ||(Wx + b) - y ||_2^2 $$

                Args:
                    input_dim: number of dimensions of the input (1+)
                    bias: If false, will not use $+b$
                    learning_rate: learning_rate for the optimizer
                    optimizer: the optimizer to use (default='Adam')
                    l1_strength: L1 regularization strength (default=None)
                    l2_strength: L2 regularization strength (default=None)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.linear = nn.Linear(in_features=self.hparams.input_dim, out_features=1, bias=bias)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        # flatten any input
        x = x.view(x.size(0), -1)

        y_hat = self(x)

        loss = F.mse_loss(y_hat, y)

        # L1 regularizer
        if self.hparams.l1_strength is not None:
            l1_reg = torch.Tensor(0.)
            for param in self.parameters():
                l1_reg += torch.norm(param, 1)
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength is not None:
            l2_reg = torch.Tensor(0.)
            for param in self.parameters():
                l2_reg += torch.norm(param, 2)
            loss += self.hparams.l2_strength * l2_reg

        tensorboard_logs = {'train_mse_loss': loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'loss': loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        return {'val_loss': F.mse_loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_mse_loss': val_loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'val_loss': val_loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.mse_loss(y_hat, y)}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_mse_loss': test_loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'test_loss': test_loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--input_dim', type=int, default=None)
        parser.add_argument('--bias', default='store_true')
        parser.add_argument('--batch_size', type=int, default=16)
        return parser


def train():
    pl.seed_everything(1234)
    x, y = load_data()

    parser = ArgumentParser()
    parser = LinearRegression.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    train_loader = DataLoader(SklearnDataSet(x, y, shuffle=True, random_state=1234), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SklearnDataSet(x, y, shuffle=True, random_state=1234), batch_size=args.batch_size)

    # model
    model = LinearRegression(input_dim=13, l1_strength=1, l2_strength=1)

    # train
    trainer = pl.Trainer.from_argparse_args(args, max_epochs=30)

    trainer.fit(model, train_loader, val_loader)
    print(model.linear.weight)
    print(model.linear.bias)


if __name__ == '__main__':
    # load_data()
    # show()
    train()
    print('aa')
