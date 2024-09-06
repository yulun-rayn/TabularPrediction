import time
import math

from hyperopt import hp

from sklearn.linear_model import LogisticRegression, Ridge

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

MULTITHREAD = -1

param_grid = {
    'max_iter': hp.randint('max_iter', 50, 500),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'C': hp.loguniform('C', math.log(1e-1), math.log(1e6))
}

def ridge_predict(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300, no_tune=None):
    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=True, impute=True, standardize=True, cat_features=cat_features)

    def model_(**params):
        if is_classification(metric_used):
            return LogisticRegression(penalty='l2', solver='saga', tol=1e-4, n_jobs=MULTITHREAD, **params)
        else:
            return Ridge(tol=1e-4, **params)

    start_time = time.time()
    pred, _ = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune)
    end_time = time.time()
    return test_y, pred, end_time-start_time
