import time
import math

import numpy as np

from hyperopt import hp

import torch
import torch.nn.functional as F

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

param_grid = {
    'dim': hp.choice('dim', [32, 64, 128, 256]),
    'depth': hp.choice('depth', [2, 3, 6, 12]),
    'heads': hp.choice('heads', [2, 4, 8]),
    'dropout': hp.choice('dropout', [0, 0.2, 0.4, 0.6, 0.8]),
}

def get_scoring_string(metric_used):
    if metric_used.__name__ == "cross_entropy_metric":
        return 'classification'
    elif metric_used.__name__ == "mse_metric":
        return 'regression'
    else:
        raise Exception('No scoring string found for metric')
