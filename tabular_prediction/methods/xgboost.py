import time
import math

import numpy as np

from hyperopt import hp

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

MULTITHREAD = -1

param_grid = {
    'learning_rate': hp.loguniform('learning_rate', -7, math.log(1)),
    'max_depth': hp.randint('max_depth', 1, 10),
    'subsample': hp.uniform('subsample', 0.2, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1),
    'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
    'alpha': hp.loguniform('alpha', -16, 2),
    'lambda': hp.loguniform('lambda', -16, 2),
    'gamma': hp.loguniform('gamma', -16, 2),
    'n_estimators': hp.randint('n_estimators', 100, 4000), # This is smaller than in paper
}

def get_scoring_string(metric_used, multiclass=True):
    if metric_used.__name__ == "cross_entropy_metric":
        if multiclass:
            return 'multi:softmax'
        else:
            return 'binary:logitraw'
    elif metric_used.__name__ == "mse_metric":
        return 'reg:squarederror'
    elif metric_used.__name__ == "mae_metric":
        return 'reg:absoluteerror'
    else:
        raise Exception('No scoring string found for metric')

def xgboost_predict(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300, no_tune=None, gpu_id=None):
    from xgboost import XGBClassifier, XGBRegressor

    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=False, impute=False, standardize=False, cat_features=cat_features)

    if gpu_id is not None:
        gpu_params = {'tree_method':'gpu_hist', 'gpu_id':gpu_id}
    else:
        gpu_params = {}

    def model_(**params):
        if is_classification(metric_used):
            return XGBClassifier(
                objective=get_scoring_string(metric_used, multiclass=(len(np.unique(y)) > 2)),
                use_label_encoder=False,
                nthread=MULTITHREAD,
                **params,
                **gpu_params)
        else:
            return XGBRegressor(
                objective=get_scoring_string(metric_used, multiclass=(len(np.unique(y)) > 2)),
                use_label_encoder=False,
                nthread=MULTITHREAD,
                **params,
                **gpu_params)

    start_time = time.time()
    summary = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune)
    end_time = time.time()
    return test_y, summary, end_time-start_time
