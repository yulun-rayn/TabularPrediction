import time

import numpy as np

from hyperopt import hp

from .resnet_lib import TabResNet

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

param_grid = {
    # 'd_token': hp.choice('d_token', [4, 8, 16]),
    # 'n_blocks': hp.choice('n_blocks', [2, 3, 4]),
    # 'd_main': hp.choice('d_main', [32, 64, 128]),
    # 'hidden_multiplier': hp.choice('hidden_multiplier', [1, 2, 4]),
    'learning_rate': hp.choice('learning_rate', [1e-4, 1e-3, 1e-2]),
    'batch_size': hp.choice('learning_rate', [64, 128, 256]),
    'epochs': hp.choice('epochs', [50, 100]),
}

def resnet_predict(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300, no_tune=None):
    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=False, impute=False, standardize=False, cat_features=cat_features)

    def model_(**params):
        return TabResNet(n_features=x.shape[1], cat_features=cat_features,
            is_classification=is_classification(metric_used),
            n_classes=len(np.unique(y)), **params)

    start_time = time.time()
    summary = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune)
    end_time = time.time()
    return test_y, summary, end_time-start_time
