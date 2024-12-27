import numpy as np

import torch
import torch.nn.functional as F

from rtdl import ResNet, CategoricalFeatureTokenizer

from tabular_prediction.methods.utils import BaseModelTorch

class TabResNet(BaseModelTorch):
    def __init__(self, d_in: int, cat_features: list = None, 
                 d_token: int = 8, n_blocks: int = 2,
                 d_main: int = 128, hidden_multiplier: int = 2, 
                 dropout_first: float = 0.25, dropout_second: float = 0.1,
                 params: dict = None): # is_classification, n_classes, learning_rate
        super().__init__(**params)
        self.cat_features = cat_features
        d_out = self.n_classes if self.is_classification else 1

        if cat_features is not None and len(cat_features) > 0:
            self.cat_tokenizer = CategoricalFeatureTokenizer(
                cat_features, d_token, False, "uniform"
            )
            self.model = ResNet.make_baseline(
                d_in=d_in + self.cat_tokenizer.n_tokens * (self.cat_tokenizer.d_token - 1),
                d_out=d_out, n_blocks=n_blocks, d_main=d_main,
                d_hidden=d_main * hidden_multiplier,
                dropout_first=dropout_first,
                dropout_second=dropout_second
            )
        else:
            self.cat_tokenizer = None
            self.model = ResNet.make_baseline(
                d_in=d_in, d_out=d_out, n_blocks=n_blocks, d_main=d_main,
                d_hidden=d_main * hidden_multiplier,
                dropout_first=dropout_first,
                dropout_second=dropout_second
            )

        self.to_device()

    def forward(self, x):
        if self.cat_features is not None and len(self.cat_features) > 0:
            num_features = [i for i in range(x.shape[1]) if not i in self.cat_features]
            x_num = x[:, num_features]
            x_cat = x[:, self.cat_features].to(torch.int)
            x_ordered = torch.cat([x_num, self.cat_tokenizer(x_cat).flatten(1, -1)], dim=1)
        else:
            x_ordered = x

        if self.is_classification:
            out = F.softmax(self.model(x_ordered), dim=1)
        else:
            out = self.model(x_ordered)

        return out

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=float)
        X_val = np.array(X_val, dtype=float)

        return super().fit(X, y, X_val, y_val)

    def predict_helper(self, X):
        X = np.array(X, dtype=float)
        return super().predict_helper(X)
