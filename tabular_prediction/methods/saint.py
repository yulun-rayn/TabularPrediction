import time

import numpy as np

import torch

from hyperopt import hp

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

param_grid = {
    'dim': hp.choice('dim', [32, 64, 128]), #,256
    'depth': hp.choice('depth', [2, 3, 6]), #,12
    'heads': hp.choice('heads', [2, 4, 8]),
    'dropout': hp.choice('dropout', [0, 0.2, 0.4, 0.6, 0.8]),
    'epochs': hp.choice('epochs', [50, 100]),
}

def saint_predict(x, y, test_x, test_y, metric_used, cat_features=None, max_features=100, max_time=300, no_tune=None, gpu_id=0, run_id=""):
    from .saint_lib import SAINT

    if x.shape[1] > max_features:
        if cat_features is not None:
            cat_inds = torch.zeros(x.shape[1])
            cat_inds[cat_features] = 1
        sample_ids = np.random.choice(x.shape[1], max_features, replace=False)
        x = x[:, sample_ids]
        test_x = test_x[:, sample_ids]
        cat_inds = cat_inds[sample_ids]
        cat_features = torch.where(cat_inds == 1)[0]

    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=False, impute=False, standardize=False, cat_features=cat_features)

    cat_features_min = np.concatenate((x, test_x), axis=0)[:, cat_features].min(0)
    x[:, cat_features] = x[:, cat_features] - cat_features_min
    test_x[:, cat_features] = test_x[:, cat_features] - cat_features_min

    def model_(**params):
        return SAINT(
            n_features=x.shape[1],
            cat_features=cat_features,
            cat_dims=[len(np.unique(x[:, c])) for c in cat_features],
            is_classification=is_classification(metric_used),
            n_classes=len(np.unique(y)),
            run_id=run_id,
            gpu_id=gpu_id,
            **params
        )

    start_time = time.time()
    summary = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune,
        sgd=True, cv=False, run_default=False)
    end_time = time.time()
    return test_y, summary, end_time-start_time
