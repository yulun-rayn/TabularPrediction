import time
import math

import numpy as np

from hyperopt import hp

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

param_grid = {
    'num_leaves': hp.randint('num_leaves', 5, 50)
    , 'max_depth': hp.randint('max_depth', 3, 20)
    , 'learning_rate': hp.loguniform('learning_rate', -3, math.log(1.0))
    , 'n_estimators': hp.randint('n_estimators', 50, 2000)
    #, 'feature_fraction': 0.8,
    #, 'subsample': 0.2
    , 'min_child_weight': hp.choice('min_child_weight', [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4])
    , 'subsample': hp.uniform('subsample', 0.2, 0.8)
    , 'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 0.8)
    , 'reg_alpha': hp.choice('reg_alpha', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100])
    , 'reg_lambda': hp.choice('reg_lambda', [0, 1e-1, 1, 5, 10, 20, 50, 100])
}  # 'normalize': [False],

def get_scoring_string(metric_used, multiclass=True):
    if metric_used.__name__ == "cross_entropy_metric":
        if multiclass:
            return 'multiclass'
        else:
            return 'binary'
    elif metric_used.__name__ == "mse_metric":
        return 'regression'
    else:
        raise Exception('No scoring string found for metric')

def lightgbm_predict(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300, no_tune=None):
    from lightgbm import LGBMClassifier, LGBMRegressor

    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=False, impute=False, standardize=False, cat_features=cat_features)

    # Negative values in categorical features must be converted to non-negative
    cat_features_min = np.nanmin(np.concatenate((x, test_x), axis=0)[:, cat_features], axis=0)
    x[:, cat_features] = x[:, cat_features] - cat_features_min
    test_x[:, cat_features] = test_x[:, cat_features] - cat_features_min

    def model_(**params):
        if is_classification(metric_used):
            return LGBMClassifier(
                objective=get_scoring_string(metric_used, multiclass=len(np.unique(y)) > 2),
                categorical_feature=cat_features,
                use_missing=True,
                **params)
        else:
            return LGBMRegressor(
                objective=get_scoring_string(metric_used, multiclass=len(np.unique(y)) > 2),
                categorical_feature=cat_features,
                use_missing=True,
                **params)

    start_time = time.time()
    summary = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune)
    end_time = time.time()
    return test_y, summary, end_time-start_time
