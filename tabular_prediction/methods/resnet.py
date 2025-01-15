import time

import numpy as np

from hyperopt import hp

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f_deep

param_grid = {
    # 'd_token': hp.choice('d_token', [4, 8, 16]),
    # 'n_blocks': hp.choice('n_blocks', [2, 3, 4]),
    # 'd_main': hp.choice('d_main', [32, 64, 128]),
    # 'c_hidden': hp.choice('c_hidden', [1, 2, 4]),
    'learning_rate': hp.choice('learning_rate', [1e-4, 1e-3, 1e-2]),
    'batch_size': hp.choice('batch_size', [64, 128, 256]),
    'epochs': hp.choice('epochs', [50, 100]),
}

def resnet_predict(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300, no_tune=None, gpu_id=0, save_dir="output/SAINT"):
    from .resnet_lib import TabResNet

    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=False, impute=False, standardize=False, cat_features=cat_features)

    # Negative values in categorical features must be converted to non-negative
    cat_features_min = np.concatenate((x, test_x), axis=0)[:, cat_features].min(0)
    x[:, cat_features] = x[:, cat_features] - cat_features_min
    test_x[:, cat_features] = test_x[:, cat_features] - cat_features_min

    def model_(**params):
        return TabResNet(
            n_features=x.shape[1],
            cat_features=cat_features,
            cat_dims=[len(np.unique(x[:, c])) for c in cat_features],
            is_classification=is_classification(metric_used),
            n_classes=len(np.unique(y)),
            save_dir=save_dir,
            gpu_id=gpu_id,
            **params
        )

    start_time = time.time()
    summary = eval_complete_f_deep(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune)
    end_time = time.time()
    return test_y, summary, end_time-start_time
