import time
import typing as tp

import numpy as np

from sklearn.base import BaseEstimator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .models.pretrainmodel import SAINTModel
from .data_openml import DataSetCatCon
from .augmentations import embed_data_mask

from tabular_prediction.methods.utils import BaseModelTorch


class SAINT(BaseModelTorch, BaseEstimator):
    """Interface for SAINT model.
    Notes
    -----
    Specify all the parameters that can be set at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    # TabZilla: add default number of epochs.
    # default_epochs = 100  # from SAINT paper. this is equal to our max-epochs

    def __init__(self, n_features: int, cat_features: list = None, cat_dims: list = None,
                 dim: int = 64, depth: int = 3, heads: int = 4, dropout: float = 0.5,
                 is_classification: bool = None, n_classes: int = None,
                 gpu_id: tp.Union[list, int] = 0, data_parallel: bool = False,
                 learning_rate: float = 1e-3, epochs: int = 50,
                 batch_size: int = 128, val_batch_size: int = 512,
                 early_stopping_rounds: int = 5, run_id: str = "",
                 save_dir: str = None, sub_dir: str = None):
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
            save_dir=save_dir,
            sub_dir=sub_dir
        )
        self.n_features = n_features
        self.cat_features = cat_features if cat_features is not None else []
        self.cat_dims = cat_dims
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout

        if len(self.cat_features) > 0:
            num_features = [i for i in range(n_features) if not i in self.cat_features]
            # Appending 1 for CLS token, this is later used to generate embeddings.
            cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)
        else:
            num_features = list(range(n_features))
            cat_dims = np.array([1])

        # Decreasing some hyperparameter to cope with memory issues
        dim = dim if n_features < 50 else 8
        self.batch_size = self.batch_size if n_features < 50 else 64

        print("Using dim %d, depth %d and batch size %d" % (dim, depth, self.batch_size))

        self.model = SAINTModel(
            categories=tuple(cat_dims),
            num_continuous=len(num_features),
            dim=dim,
            dim_out=1,
            depth=depth,  # 6
            heads=heads,  # 8
            attn_dropout=dropout,  # 0.1
            ff_dropout=dropout,  # 0.1
            mlp_hidden_mults=(4, 2),
            cont_embeddings="MLP",
            attentiontype="colrow",
            final_mlp_style="sep",
            y_dim=self.n_classes if self.is_classification else 1,
        )

        self.to_device()

    def fit(self, X, y, X_val=None, y_val=None, r_val=0.1, time_limit=600):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        if X_val is None:
            perm = np.random.permutation(X.shape[0])
            val_size = int(r_val * X.shape[0])
            X_val = X[perm[:val_size]]
            X = X[perm[val_size:]]
            y_val = y[perm[:val_size]]
            y = y[perm[val_size:]]

        if self.is_classification is None:
            self.is_classification = not torch.is_floating_point(y)

        if self.is_classification:
            task = 'classification'
            criterion = nn.CrossEntropyLoss()
        else:
            task = 'regression'
            criterion = nn.MSELoss()

        # SAINT wants it like this...
        X = {"data": X, "mask": np.ones_like(X)}
        y = {"data": y.reshape(-1, 1)}
        X_val = {"data": X_val, "mask": np.ones_like(X_val)}
        y_val = {"data": y_val.reshape(-1, 1)}

        train_ds = DataSetCatCon(X, y, self.cat_features, task)
        trainloader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

        val_ds = DataSetCatCon(X_val, y_val, self.cat_features, task)
        valloader = DataLoader(
            val_ds, batch_size=self.val_batch_size, shuffle=True, num_workers=2
        )

        min_val_loss = float("inf")
        min_val_loss_idx = 0
        self.save_model(filename_extension="best")

        loss_history = []
        val_loss_history = []
        start_time = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            for i, data in enumerate(trainloader, 0):
                # x_categ is the the categorical data,
                # x_cont has continuous data,
                # y_gts has ground truth ys.
                # cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS
                # token) set to 0s.
                # con_mask is an array of ones same shape as x_cont.
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                y_gts = y_gts.squeeze(-1).to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                # We are converting the data to embeddings in the next step
                _, x_categ_enc, x_cont_enc = embed_data_mask(
                    x_categ, x_cont, cat_mask, con_mask, self.model
                )

                reps = self.model.transformer(x_categ_enc, x_cont_enc)

                # select only the representations corresponding to CLS token
                # and apply mlp on it in the next step to get the predictions.
                y_reps = reps[:, 0, :]

                y_outs = self.model.mlpfory(y_reps)

                if not self.is_classification:
                    # y_outs = y_outs.squeeze()
                    y_outs = y_outs.reshape((y_gts.shape[0], ))

                loss = criterion(y_outs, y_gts)
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early Stopping
            val_loss = 0.0
            val_dim = 0
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(valloader):
                    x_categ, x_cont, y_gts, cat_mask, con_mask = data

                    x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                    y_gts = y_gts.squeeze(-1).to(self.device)
                    cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                    _, x_categ_enc, x_cont_enc = embed_data_mask(
                        x_categ, x_cont, cat_mask, con_mask, self.model
                    )
                    reps = self.model.transformer(x_categ_enc, x_cont_enc)
                    y_reps = reps[:, 0, :]
                    y_outs = self.model.mlpfory(y_reps)

                    if not self.is_classification:
                        # y_outs = y_outs.squeeze()
                        y_outs = y_outs.reshape((y_gts.shape[0], ))

                    val_loss += criterion(y_outs, y_gts)
                    val_dim += 1

            val_loss /= val_dim
            val_loss_history.append(val_loss.item())

            print("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="best")

            if min_val_loss_idx + self.early_stopping_rounds < epoch:
                print(
                    "Validation loss has not improved for %d epochs!"
                    % self.early_stopping_rounds
                )
                print("Early stopping applies.")
                break

            runtime = time.time() - start_time
            if runtime > time_limit:
                print(
                    f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                )
                break

        # Load best model
        self.load_model(filename_extension="best")
        return loss_history, val_loss_history

    def predict_helper(self, X):
        X = {"data": X, "mask": np.ones_like(X)}
        y = {"data": np.ones((X["data"].shape[0], 1))}

        test_ds = DataSetCatCon(X, y, self.cat_features)
        testloader = DataLoader(
            test_ds, batch_size=self.val_batch_size, shuffle=False, num_workers=2
        )

        self.model.eval()

        predictions = []
        with torch.no_grad():
            for data in testloader:
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                _, x_categ_enc, x_cont_enc = embed_data_mask(
                    x_categ, x_cont, cat_mask, con_mask, self.model
                )
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = self.model.mlpfory(y_reps)

                predictions.append(y_outs.detach().cpu().numpy())
        return np.concatenate(predictions)

    def attribute(self, X, y, strategy=""):
        """Generate feature attributions for the model input.
        Two strategies are supported: default ("") or "diag". The default strategie takes the sum
        over a column of the attention map, while "diag" returns only the diagonal (feature attention to itself)
        of the attention map.
        return array with the same shape as X.
        """
        global my_attention
        self.load_model(filename_extension="best")

        X = {"data": X, "mask": np.ones_like(X)}
        y = {"data": np.ones((X["data"].shape[0], 1))}

        test_ds = DataSetCatCon(X, y, self.cat_features)
        testloader = DataLoader(
            test_ds, batch_size=self.val_batch_size, shuffle=False, num_workers=2
        )

        self.model.eval()
        # print(self.model)
        # Apply hook.
        my_attention = torch.zeros(0)

        def sample_attribution(layer, minput, output):
            global my_attention
            # print(minput)
            """ an hook to extract the attention maps. """
            h = layer.heads
            q, k, v = layer.to_qkv(minput[0]).chunk(3, dim=-1)
            q, k, v = map(
                lambda t: t.reshape(*t.shape[:2], h, -1).permute(0, 2, 1, 3), (q, k, v)
            )
            sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * layer.scale
            my_attention = sim.softmax(dim=-1)

        # print(type(self.model.transformer.layers[0][0].fn.fn))
        self.model.transformer.layers[0][0].fn.fn.register_forward_hook(
            sample_attribution
        )
        attributions = []
        with torch.no_grad():
            for data in testloader:
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)
                # print(x_categ.shape, x_cont.shape)
                _, x_categ_enc, x_cont_enc = embed_data_mask(
                    x_categ, x_cont, cat_mask, con_mask, self.model
                )
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                # y_reps = reps[:, 0, :]
                # y_outs = self.model.mlpfory(y_reps)
                if strategy == "diag":
                    attributions.append(
                        my_attention.sum(dim=1)[:, 1:, 1:].diagonal(0, 1, 2)
                    )
                else:
                    attributions.append(my_attention.sum(dim=1)[:, 1:, 1:].sum(dim=1))

        attributions = np.concatenate(attributions)
        return attributions
