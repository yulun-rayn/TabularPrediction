import time

from hyperopt import hp

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

MULTITHREAD = -1

param_grid = {
    'n_estimators': hp.randint('n_estimators', 20, 200),
    'max_features': hp.choice('max_features', ['auto', 'sqrt']),
    'max_depth': hp.randint('max_depth', 1, 45),
    'min_samples_split': hp.choice('min_samples_split', [5, 10])
}

def randomforest_predict(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300, no_tune=None):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=False, impute=True, standardize=False, cat_features=cat_features)

    def model_(**params):
        if is_classification(metric_used):
            return RandomForestClassifier(n_jobs=MULTITHREAD, **params)
        else:
            return RandomForestRegressor(n_jobs=MULTITHREAD, **params)

    start_time = time.time()
    summary = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune)
    end_time = time.time()
    return test_y, summary, end_time-start_time
