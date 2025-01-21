import time

import numpy as np

from hyperopt import hp

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

param_grid = {
    'C': hp.choice('C', [0.1, 1, 10, 100]),
    'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid'])
}

def svm_predict(x, y, test_x, test_y, metric_used, cat_features=None, max_train=5000, max_time=300, no_tune=None):
    from sklearn.svm import SVC, SVR

    if x.shape[0] > max_train:
        sample_ids = np.random.choice(x.shape[0], max_train, replace=False)
        x = x[sample_ids]
        y = y[sample_ids]

    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=True, impute=True, standardize=True, cat_features=cat_features)

    def model_(**params):
        if is_classification(metric_used):
            return SVC(probability=True, gamma="auto", **params)
        else:
            return SVR(gamma="auto", **params)

    start_time = time.time()
    summary = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune)
    end_time = time.time()
    return test_y, summary, end_time-start_time
