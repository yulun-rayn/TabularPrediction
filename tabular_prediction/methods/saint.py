import time
import math

import numpy as np

from hyperopt import hp

from .saint_lib import SAINT

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

def saint_predict(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300, no_tune=None):
    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=False, impute=False, standardize=False, cat_features=cat_features)

    def model_(**params):
        return SAINT(n_features=x.shape[1], cat_features=cat_features,
            cat_dims=[np.unique(x[:, c]) for c in cat_features],
            is_classification=is_classification(metric_used),
            n_classes=len(np.unique(y)), **params)

    start_time = time.time()
    summary = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune)
    end_time = time.time()
    return test_y, summary, end_time-start_time
