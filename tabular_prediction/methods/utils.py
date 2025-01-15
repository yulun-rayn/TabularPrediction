import os
import math
import time
import string
import random
import typing as tp
from datetime import datetime

import numpy as np
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def get_output_path(dir_path, filename, file_type, directory=None, extension=None):
    if directory:
        # For example: .../models
        dir_path = os.path.join(dir_path, directory)

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, filename)

    if extension is not None:
        file_path += "_" + str(extension)

    file_path += "." + file_type

    # For example: .../m_3.pkl

    return file_path


class BaseModel:
    """Basic interface for all models.

    All implemented models should inherit from this base class to provide a common interface.

    Notes
    -----
    Specify all the parameters that can be set at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    Methods
    -------
    __init__(params, args):
        Defines the model architecture, depending on the hyperparameters (params) and command line arguments (args).
    fit(X, y, X_val=None, y_val=None)
        Trains the model on the trainings dataset (X, y). Validates the training process and uses early stopping
        if a validation set (X_val, y_val) is provided. Returns the loss history and validation loss history.
    predict(X)
        Predicts the labels of the test dataset (X). Saves and returns the predictions.
    attribute(X, y)
        Extract feature attributions for input pair (X, y)
    clone()
        Creates a fresh copy of the model using the same parameters, but ignoring any trained weights. This method
        is necessary for the cross validation.
    """

    # this list should be populated with "classification" and "regression" for each subclass if 
    # the model is not implemented for these objective types
    objtype_not_implemented = []

    def __init__(self, is_classification: bool, n_classes: int = None):
        """Defines the model architecture.

        After calling this method, self.model has to be defined.
        """
        self.is_classification = is_classification
        self.n_classes = n_classes

        # Model definition has to be implemented by the concrete model
        self.model = None

        # Create a placeholder for the predictions on the test dataset
        self.predictions = None
        self.prediction_probabilities = (
            None  # Only used by classification
        )

    # added for TabZilla bookkeeping
    def get_metadata(self):
        return {
            "name": self.__class__.__name__,
            "is_classification": self.is_classification,
            "n_classes": self.n_classes,
        }

    def get_classes(self):
        if "classes_" not in dir(self):
            return None
        return self.classes_

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: tp.Union[None, np.ndarray] = None,
        y_val: tp.Union[None, np.ndarray] = None,
    ) -> tp.Tuple[list, list]:
        """Trains the model.

        The training is done on the trainings dataset (X, y). If a validation set (X_val, y_val) is provided,
        the model state is validated during the training, to allow early stopping.

        Returns the loss history and validation loss history if the loss and validation loss development during
        the training are logged. Otherwise empty lists are returned.

        :param X: trainings data
        :param y: labels of trainings data
        :param X_val: validation data
        :param y_val: labels of validation data
        :return: loss history, validation loss history
        """

        self.model.fit(X, y)

        # Should return loss history and validation loss history
        return [], []

    # Patch around the original predict method to handle case of missing classes in training set and subsampling.
    # This needs to be done as a separate method, since several of the inheriting classes override the predict_proba or
    # predict methods.
    def predict_wrapper(self, X: np.ndarray, max_rows : int) -> tp.Tuple[np.ndarray, np.ndarray]:
        if max_rows > 0 and X.shape[0] > max_rows:
            X_ens = []
            X_preds = []
            X_probas = []
            for idx, i in enumerate(range(0, X.shape[0], max_rows)):
                print(f"Fitting samples {idx+1} of {math.ceil(X.shape[0]/max_rows)}")
                X_ens.append(X[i:i+max_rows])
                preds, probas = self.predict(X_ens[-1])
                X_preds.append(preds)
                X_probas.append(probas)
            self.predictions, self.prediction_probabilities = np.concatenate(X_preds, axis=0), np.concatenate(X_probas, axis=0)
        else:
            self.predictions, self.prediction_probabilities = self.predict(X)
        if (
            self.is_classification
            and self.prediction_probabilities.shape[1] != self.n_classes
        ):
            # Handle special case of missing classes in training set, which can (depending on the model)  result in
            # predictions only being made for those classes
            classes_ = self.get_classes()
            if classes_ is None:
                raise NotImplementedError(
                    f"Cannot infer classes for model of type {type(self)}"
                )
            # From https://github.com/scikit-learn/scikit-learn/issues/21568#issuecomment-984030911
            y_score_expanded = np.zeros(
                (self.prediction_probabilities.shape[0], self.n_classes),
                dtype=self.prediction_probabilities.dtype,
            )
            for idx, class_id in enumerate(classes_):
                y_score_expanded[:, class_id] = self.prediction_probabilities[:, idx]
            self.prediction_probabilities = y_score_expanded
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions, self.prediction_probabilities

    def predict(self, X: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Returns the regression value or the concrete classes of classification tasks.
        (Save predictions to self.predictions)

        :param X: test data
        :return: predicted values / classes of test data (Shape N x 1)
        """

        # TabZilla update: always return prediction probabilities
        self.prediction_probabilities = np.array([])

        if self.is_classification:
            self.prediction_probabilities = self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)
        else:
            self.predictions = self.model.predict(X)

        return self.predictions, self.prediction_probabilities

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Only implemented for classification tasks.
        Returns the probability distribution over the classes C.
        (Save probabilities to self.prediction_probabilities)

        :param X: test data
        :return: probabilities for the classes (Shape N x C)
        """

        self.prediction_probabilities = self.model.predict_proba(X)
        return self.prediction_probabilities

    def clone(self):
        """Clone the model.

        Creates a fresh copy of the model using the same parameters, but ignoring any trained weights. This method
        is necessary for the cross validation.

        :return: Copy of the current model without trained parameters
        """
        return self.__class__(is_classification=self.is_classification, n_classes=self.n_classes)

    # TabZilla: add placeholder methods for get_random_parameters() and default_parameters()
    @classmethod
    def get_random_parameters(cls, seed: int):
        """
        returns a random set of hyperparameters, which can be replicated using the provided seed
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    def default_parameters(cls):
        """
        returns the default set of hyperparameters
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    def get_model_size(self):
        raise NotImplementedError(
            "Calculation of model size has not been implemented for this model."
        )

    def attribute(cls, X: np.ndarray, y: np.ndarray, strategy: str = "") -> np.ndarray:
        """Get feature attributions for inherently interpretable models. This function is only implemented for
        interpretable models.

        :param X: data (Shape N x D)
        :param y: labels (Shape N) for which the attribution should be computed for (
        usage of these labels depends on the specific model)

        :strategy: if there are different strategies that can be used to compute the attributions they can be passed
        here. Passing an empty sting should always result in the default strategy.

        :return The (non-normalized) importance attributions for each feature in each data point. (Shape N x D)
        """
        raise NotImplementedError(
            f"This method is not implemented for class {type(cls)}."
        )


class BaseModelTorch(BaseModel):
    """Basic interface for Torch models.
    Notes
    -----
    Specify all the parameters that can be set at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """
    def __init__(self, is_classification: bool = None, n_classes: int = None,
                 gpu_id: tp.Union[list, int] = 0, data_parallel: bool = False,
                 learning_rate: float = 1e-3, epochs: int = 50,
                 batch_size: int = 128, val_batch_size: int = 512,
                 early_stopping_rounds: int = 5, run_id: str = "",
                 save_dir: str = None, sub_dir: str = None):
        super().__init__(is_classification=is_classification, n_classes=n_classes)
        if gpu_id is not None and not isinstance(gpu_id, list):
            gpu_id = [gpu_id]
        self.gpu_id = gpu_id
        self.data_parallel = data_parallel
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.run_id = run_id

        self.device = self.get_device()

        if save_dir is None or save_dir == "":
            save_dir = os.path.join("output/", self.__class__.__name__, run_id)
        self.save_dir = save_dir
        if sub_dir is None or sub_dir == "":
            sub_dir = "tmp_" + datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
        self.sub_dir = sub_dir

    def to_device(self):
        if self.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_id)

        print("On Device:", self.device)
        self.model.to(self.device)

    def get_device(self):
        if self.gpu_id is not None and torch.cuda.is_available():
            if self.data_parallel:
                device = "cuda"
            else:
                device = f"cuda:{str(self.gpu_id[0])}"
        else:
            device = "cpu"

        return torch.device(device)

    def forward(self, X):
        return self.model(X)

    # TabZilla: added a time limit
    def fit(self, X, y, X_val=None, y_val=None, r_val=0.1, time_limit=600):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        X = torch.as_tensor(X, dtype=torch.float)
        y = torch.as_tensor(y)
        if X_val is None:
            perm = torch.randperm(X.shape[0])
            val_size = int(r_val * X.shape[0])
            X_val = X[perm[:val_size]]
            X = X[perm[val_size:]]
            y_val = y[perm[:val_size]]
            y = y[perm[val_size:]]
        X_val = torch.as_tensor(X_val, dtype=torch.float)
        y_val = torch.as_tensor(y_val)

        if self.is_classification is None:
            self.is_classification = not torch.is_floating_point(y)

        if self.is_classification:
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.val_batch_size, shuffle=True
        )

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        start_time = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            for i, (batch_X, batch_y) in enumerate(train_loader):

                out = self.forward(batch_X.to(self.device))

                if not self.is_classification:
                    # out = out.squeeze()
                    out = out.reshape((batch_y.shape[0], ))

                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early Stopping
            val_loss = 0.0
            val_dim = 0
            self.model.eval()
            for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                out = self.forward(batch_val_X.to(self.device))

                if not self.is_classification:
                    #out = out.squeeze()
                    out = out.reshape((batch_val_X.shape[0], ))

                val_loss += loss_func(out, batch_val_y.to(self.device))
                val_dim += 1

            val_loss /= val_dim
            val_loss_history.append(val_loss.item())

            print("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="best", directory=self.directory)
                
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

    def predict(self, X):
        # tabzilla update: return prediction probabilities
        if self.is_classification:
            probs = self.predict_proba(X)
            self.predictions = np.argmax(probs, axis=1)
        else:
            self.predictions = self.predict_helper(X)
            probs = np.array([])

        return self.predictions, probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_helper(X)
        probas = softmax(logits, axis=1)

        self.prediction_probabilities = probas
        return self.prediction_probabilities

    def predict_helper(self, X):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=2,
        )
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                preds = self.forward(batch_X[0].to(self.device))

                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)

    def save_model(self, filename_extension="", save_dir=None, sub_dir=None):
        self.state_dict_path = get_output_path(
            save_dir if save_dir is not None else self.save_dir,
            directory=sub_dir if sub_dir is not None else self.sub_dir,
            filename="m",
            extension=filename_extension,
            file_type="pt",
        )
        torch.save(self.model.state_dict(), self.state_dict_path)

    def load_model(self, filename_extension=None, save_dir=None, sub_dir=None):
        if filename_extension is None:
            state_dict_path = self.state_dict_path
        else:
            state_dict_path = get_output_path(
                save_dir if save_dir is not None else self.save_dir,
                directory=sub_dir if sub_dir is not None else self.sub_dir,
                filename="m",
                extension=filename_extension,
                file_type="pt",
            )
        state_dict = torch.load(state_dict_path)
        assert len(state_dict.keys()) > 0, f"Error! state_dict we loaded had no keys: {state_dict}"
        self.model.load_state_dict(state_dict)

    def get_model_size(self):
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size
