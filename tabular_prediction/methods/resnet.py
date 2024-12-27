import time
import math

import numpy as np

from hyperopt import hp

from .resnet_lib import TabResNet

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

param_grid = {
    # 'd_token': hp.choice('d_token', [4, 8, 16]),
    # 'n_blocks': hp.choice('n_blocks', [2, 3, 4]),
    # 'd_main': hp.choice('d_main', [32, 64, 128]),
    # 'hidden_multiplier': hp.choice('hidden_multiplier', [1, 2, 4]),
    'learning_rate': hp.choice('learning_rate', [1e-4, 1e-3, 1e-2])
}

def get_scoring_string(metric_used):
    if metric_used.__name__ == "cross_entropy_metric":
        return 'classification'
    elif metric_used.__name__ == "mse_metric":
        return 'regression'
    else:
        raise Exception('No scoring string found for metric')

