import time

from hyperopt import hp

from sklearn.svm import SVC, SVR

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

param_grid = {
    'C': hp.choice('C', [0.1, 1, 10, 100]),
    'gamma': hp.choice('gamma', ['auto', 'scale']),
    'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid'])
}

def svm_metric(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300, no_tune=None):
    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=True, impute=True, standardize=True, cat_features=cat_features)

    def model_(**params):
        if is_classification(metric_used):
            return SVC(probability=True, **params)
        else:
            return SVR(**params)

    start_time = time.time()
    pred, _ = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune)
    end_time = time.time()
    return test_y, pred, end_time-start_time
