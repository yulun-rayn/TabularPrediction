import numpy as np
import typing as tp

from sklearn.base import BaseEstimator

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtdl import ResNet, CategoricalFeatureTokenizer

from tabular_prediction.methods.utils import BaseModelTorch

class TabResNet(BaseModelTorch, BaseEstimator):
    """Interface for Tabular ResNet model.
    Notes
    -----
    Specify all the parameters that can be set at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """
    def __init__(self, n_features: int, cat_features: list = None, cat_dims: list = None,
                 d_token: int = 8, n_blocks: int = 2, d_main: int = 128, c_hidden: int = 2,
                 dropout_first: float = 0.25, dropout_second: float = 0.1,
                 is_classification: bool = None, n_classes: int = None,
                 gpu_id: tp.Union[list, int] = 0, data_parallel: bool = False,
                 learning_rate: float = 1e-3, epochs: int = 50,
                 batch_size: int = 128, val_batch_size: int = 512,
                 early_stopping_rounds: int = 5, run_id: str = "",
                 directory: str = None):
        super().__init__(
            is_classification=is_classification,
            n_classes=n_classes,
            gpu_id=gpu_id,
            data_parallel=data_parallel,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            early_stopping_rounds=early_stopping_rounds,
            run_id=run_id,
            directory=directory
        )
        self.n_features = n_features
        self.cat_features = cat_features if cat_features is not None else []
        self.cat_dims = cat_dims
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.d_main = d_main
        self.c_hidden = c_hidden
        self.dropout_first = dropout_first
        self.dropout_second = dropout_second

        if len(self.cat_features) > 0:
            self.num_features = [i for i in range(n_features) if not i in self.cat_features]
            self.cat_tokenizer = CategoricalFeatureTokenizer(cat_dims, d_token, False, "uniform")
            self.model = ResNet.make_baseline(
                d_in=n_features + self.cat_tokenizer.n_tokens * (self.cat_tokenizer.d_token - 1),
                d_out=self.n_classes if self.is_classification else 1,
                n_blocks=n_blocks, d_main=d_main, d_hidden=d_main * c_hidden,
                dropout_first=dropout_first, dropout_second=dropout_second
            )
        else:
            self.num_features = list(range(n_features))
            self.cat_tokenizer = None
            self.model = ResNet.make_baseline(
                d_in=n_features, d_out=self.n_classes if self.is_classification else 1,
                n_blocks=n_blocks, d_main=d_main, d_hidden=d_main * c_hidden,
                dropout_first=dropout_first, dropout_second=dropout_second
            )

        self.to_device()

    def to_device(self):
        if self.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_id)
            self.cat_tokenizer = nn.DataParallel(self.cat_tokenizer, device_ids=self.gpu_id)

        print("On Device:", self.device)
        self.model.to(self.device)
        if len(self.cat_features) > 0:
            self.cat_tokenizer.to(self.device)

    def forward(self, x):
        if len(self.cat_features) > 0:
            x_num = x[:, self.num_features]
            x_cat = x[:, self.cat_features].to(torch.int)
            x_ordered = torch.cat([x_num, self.cat_tokenizer(x_cat).flatten(1, -1)], dim=1)
        else:
            x_ordered = x

        return self.model(x_ordered)

    def fit(self, X, y, X_val=None, y_val=None):
        return super().fit(X, y, X_val, y_val)

    def predict_helper(self, X):
        X = np.array(X, dtype=float)
        return super().predict_helper(X)
