import time

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

from tabular_prediction.utils import is_classification, preprocess_impute


def get_scoring_string(metric_used, multiclass=True):
    if metric_used.__name__ == "accuracy_metric":
        return 'accuracy'
    elif metric_used.__name__ == "cross_entropy_metric":
        return 'log_loss'
    elif metric_used.__name__ == "auc_metric":
        if multiclass:
            return 'roc_auc_ovo_macro'
        else:
            return 'roc_auc'
    elif metric_used.__name__ == "balanced_accuracy_metric":
        return 'balanced_accuracy'
    elif metric_used.__name__ == "average_precision_metric":
        return 'average_precision'
    elif metric_used.__name__ == "rmse_metric":
        return 'root_mean_squared_error'
    elif metric_used.__name__ == "mse_metric":
        return 'mean_squared_error'
    elif metric_used.__name__ == "mae_metric":
        return 'mean_absolute_error'
    elif metric_used.__name__ == "r2_metric":
        return 'r2'
    else:
        raise Exception('No scoring string found for metric')


def autogluon_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False
                                             , cat_features=cat_features
                                             , impute=False
                                             , standardize=False)
    train_data = pd.DataFrame(np.concatenate([x, y[:, np.newaxis]], 1))
    test_data = pd.DataFrame(np.concatenate([test_x, test_y[:, np.newaxis]], 1))
    if is_classification(metric_used):
        problem_type = 'multiclass' if len(np.unique(y)) > 2 else 'binary'
    else:
        problem_type = 'regression'
    # AutoGluon automatically infers datatypes, we don't specify the categorical labels
    start_time = time.time()
    predictor = TabularPredictor(
        label=train_data.columns[-1],
        eval_metric=get_scoring_string(metric_used, multiclass=(len(np.unique(y)) > 2)),
        problem_type=problem_type
        ## seed=int(y[:].sum()) doesn't accept seed
    ).fit(
        train_data=train_data,
        time_limit=max_time,
        presets=['best_quality']
        # The seed is deterministic but varies for each dataset and each split of it
    )

    if is_classification(metric_used):
        pred = predictor.predict_proba(test_data, as_multiclass=True).values
    else:
        pred = predictor.predict(test_data).values
    end_time = time.time()

    return test_y, pred, end_time-start_time
