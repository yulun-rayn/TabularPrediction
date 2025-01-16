import time

from hyperopt import hp

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

MULTITHREAD = -1

param_grid = {
    'n_neighbors': hp.randint('n_neighbors', 1, 16)
}

def knn_predict(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300, no_tune=None):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=True, impute=True, standardize=True, cat_features=cat_features)

    def model_(**params):
        if is_classification(metric_used):
            return KNeighborsClassifier(n_jobs=MULTITHREAD, **params)
        else:
            return KNeighborsRegressor(n_jobs=MULTITHREAD, **params)

    start_time = time.time()
    summary = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune)
    end_time = time.time()
    return test_y, summary, end_time-start_time
