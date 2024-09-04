import time
import math

import numpy as np

from hyperopt import hp

from catboost import CatBoostClassifier, CatBoostRegressor

from tabular_prediction.utils import is_classification, make_pd_from_np, preprocess_impute, eval_complete_f

MULTITHREAD = -1

param_grid = {
    'learning_rate': hp.loguniform('learning_rate', math.log(math.pow(math.e, -5)), math.log(1)),
    'random_strength': hp.randint('random_strength', 1, 20),
    'l2_leaf_reg': hp.loguniform('l2_leaf_reg', math.log(1), math.log(10)),
    'bagging_temperature': hp.uniform('bagging_temperature', 0., 1),
    'leaf_estimation_iterations': hp.randint('leaf_estimation_iterations', 1, 20),
    'iterations': hp.randint('iterations', 100, 4000), # This is smaller than in paper, 4000 leads to ram overusage
}

def get_scoring_string(metric_used):
    if metric_used.__name__ == "cross_entropy_metric":
        return 'Logloss'
    elif metric_used.__name__ == "rmse_metric":
        return 'RMSE'
    elif metric_used.__name__ == "mae_metric":
        return 'MAE'
    else:
        raise Exception('No scoring string found for metric')

def catboost_metric(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300, no_tune=None, gpu_id=None):
    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=False, impute=False, standardize=False, cat_features=cat_features)

    # Nans in categorical features must be encoded as separate class
    x[:, cat_features], test_x[:, cat_features] = (
        np.nan_to_num(x[:, cat_features], -1), np.nan_to_num(test_x[:, cat_features], -1)
    )

    if gpu_id is not None:
        gpu_params = {'task_type':"GPU", 'devices':gpu_id}
    else:
        gpu_params = {}

    x = make_pd_from_np(x, cat_features=cat_features)
    test_x = make_pd_from_np(test_x, cat_features=cat_features)

    def model_(**params):
        if is_classification(metric_used):
            return CatBoostClassifier(
                loss_function=get_scoring_string(metric_used),
                thread_count=MULTITHREAD,
                used_ram_limit='4gb',
                random_seed=int(y[:].sum()),
                logging_level='Silent',
                cat_features=cat_features,
                **gpu_params,
                **params)
        else:
            return CatBoostRegressor(
                loss_function=get_scoring_string(metric_used),
                thread_count=MULTITHREAD,
                used_ram_limit='4gb',
                random_seed=int(y[:].sum()),
                logging_level='Silent',
                cat_features=cat_features,
                **gpu_params,
                **params)

    start_time = time.time()
    pred, _ = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune)
    end_time = time.time()
    return test_y, pred, end_time-start_time
