import time

import numpy as np
import pandas as pd

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import accuracy, log_loss, balanced_accuracy, average_precision, mean_squared_error, mean_absolute_error, r2

from tabular_prediction.utils import is_classification, make_pd_from_np, preprocess_impute

MULTITHREAD = -1

def get_scoring_string(metric_used):
    if metric_used.__name__ == "accuracy_metric":
        return accuracy
    elif metric_used.__name__ == "cross_entropy_metric":
        return log_loss
    elif metric_used.__name__ == "balanced_accuracy_metric":
        return balanced_accuracy
    elif metric_used.__name__ == "average_precision_metric":
        return average_precision
    elif metric_used.__name__ == "mse_metric":
        return mean_squared_error
    elif metric_used.__name__ == "mae_metric":
        return mean_absolute_error
    elif metric_used.__name__ == "r2_metric":
        return r2
    else:
        raise Exception('No scoring string found for metric')

def autosklearn_metric(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300):
    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=False, impute=False, standardize=False, cat_features=cat_features)

    x = make_pd_from_np(x, cat_features=cat_features)
    test_x = make_pd_from_np(test_x, cat_features=cat_features)

    if is_classification(metric_used):
        model_ = AutoSklearnClassifier
    else:
        model_ = AutoSklearnRegressor

    model = model_(time_left_for_this_task=max_time,
                   memory_limit=30000,
                   n_jobs=MULTITHREAD,
                   seed=int(y[:].sum()), # The seed is deterministic but varies for each dataset and each split of it
                   metric=get_scoring_string(metric_used))

    start_time = time.time()
    model.fit(x, y)

    if is_classification(metric_used):
        pred = model.predict_proba(test_x)
    else:
        pred = model.predict(test_x)
    end_time = time.time()

    return test_y, pred, end_time-start_time
