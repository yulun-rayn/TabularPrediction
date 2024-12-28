import numpy as np

import torch
import torch.nn.functional as F

from rtdl import ResNet, CategoricalFeatureTokenizer

from tabular_prediction.methods.utils import BaseModelTorch

class TabResNet(BaseModelTorch):
    def __init__(self, n_features: int, cat_features: list = None,
                 d_token: int = 8, n_blocks: int = 2,
                 d_main: int = 128, hidden_multiplier: int = 2,
                 dropout_first: float = 0.25, dropout_second: float = 0.1,
                 is_classification: bool = None, n_classes: int = None,
                 learning_rate: float = 1e-3, epochs: int = 100,
                 batch_size: int = 64, val_batch_size: int = 512,
                 **kwargs):
        super().__init__(
            is_classification=is_classification,
            n_classes=n_classes,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            **kwargs
        )
        self.cat_features = cat_features if cat_features is not None else []
        d_out = self.n_classes if self.is_classification else 1

        if len(self.cat_features) > 0:
            self.num_features = [i for i in range(n_features) if not i in self.cat_features]
            self.cat_tokenizer = CategoricalFeatureTokenizer(
                cat_features, d_token, False, "uniform"
            )
            self.model = ResNet.make_baseline(
                d_in=n_features + self.cat_tokenizer.n_tokens * (self.cat_tokenizer.d_token - 1),
                d_out=d_out, n_blocks=n_blocks, d_main=d_main,
                d_hidden=d_main * hidden_multiplier,
                dropout_first=dropout_first,
                dropout_second=dropout_second
            )
        else:
            self.num_features = list(range(n_features))
            self.cat_tokenizer = None
            self.model = ResNet.make_baseline(
                d_in=n_features, d_out=d_out, n_blocks=n_blocks, d_main=d_main,
                d_hidden=d_main * hidden_multiplier,
                dropout_first=dropout_first,
                dropout_second=dropout_second
            )

        self.to_device()

    def forward(self, x):
        if len(self.cat_features) > 0:
            x_num = x[:, self.num_features]
            x_cat = x[:, self.cat_features].to(torch.int)
            x_ordered = torch.cat([x_num, self.cat_tokenizer(x_cat).flatten(1, -1)], dim=1)
        else:
            x_ordered = x

        return self.model(x_ordered)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=float)
        X_val = np.array(X_val, dtype=float)

        return super().fit(X, y, X_val, y_val)

    def predict_helper(self, X):
        X = np.array(X, dtype=float)
        return super().predict_helper(X)
