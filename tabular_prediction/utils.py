import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials , space_eval, rand

CV = 5

def is_classification(metric_used):
    if metric_used.__name__ in ["accuracy_metric", "cross_entropy_metric", "auc_metric", "balanced_accuracy_metric", "average_precision_metric"]:
        return True
    return False

def make_pd_from_np(x, cat_features=[]):
    data = pd.DataFrame(x)
    for c in cat_features:
        data.iloc[:, c] = data.iloc[:, c].astype('int')
    return data

def preprocess_impute(x, y, test_x, test_y, impute, one_hot, standardize, cat_features=None):
    import warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn

    x, y, test_x, test_y = x.cpu().numpy(), y.cpu().long().numpy(), test_x.cpu().numpy(), test_y.cpu().long().numpy()
    cat_features = cat_features.tolist() if cat_features is not None else []

    if impute:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(x)
        x, test_x = imp_mean.transform(x), imp_mean.transform(test_x)

    if one_hot:
        x, test_x = make_pd_from_np(x, cat_features=cat_features),  make_pd_from_np(test_x, cat_features=cat_features)
        transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_features)], remainder="passthrough")
        transformer.fit(x)
        x, test_x = transformer.transform(x), transformer.transform(test_x)

    if standardize:
        scaler = MinMaxScaler()
        scaler.fit(x)
        x, test_x = scaler.transform(x), scaler.transform(test_x)

    return x, y, test_x, test_y, cat_features

def get_scoring_string(metric_used):
    if metric_used.__name__ == "accuracy_metric":
        return 'accuracy'
    elif metric_used.__name__ == "cross_entropy_metric":
        return 'neg_log_loss'
    elif metric_used.__name__ == "auc_metric":
        return 'roc_auc_ovo'
    elif metric_used.__name__ == "balanced_accuracy_metric":
        return 'balanced_accuracy'
    elif metric_used.__name__ == "average_precision_metric":
        return 'average_precision'
    elif metric_used.__name__ == "rmse_metric":
        return 'neg_root_mean_squared_error'
    elif metric_used.__name__ == "mse_metric":
        return 'neg_mean_squared_log_error'
    elif metric_used.__name__ == "mae_metric":
        return 'neg_mean_absolute_error'
    elif metric_used.__name__ == "r2_metric":
        return 'r2'
    else:
        raise Exception('No scoring string found for metric')

def eval_f(params, clf_, x, y, metric_used):
    scores = cross_val_score(clf_(**params), x, y, cv=CV, scoring=get_scoring_string(metric_used))
    return -np.nanmean(scores)

def eval_complete_f(x, y, test_x, clf_, param_grid, metric_used, max_time, no_tune):
    start_time = time.time()
    def stop(trial):
        return time.time() - start_time > max_time, []

    if no_tune is None:
        default = eval_f({}, clf_, x, y, metric_used)
        trials = Trials()
        best = fmin(
            fn=lambda params: eval_f(params, clf_, x, y, metric_used),
            space=param_grid,
            algo=rand.suggest,
            rstate=np.random.default_rng(int(y[:].sum()) % 10000),
            early_stop_fn=stop,
            trials=trials,
            catch_eval_exceptions=True,
            verbose=True,
            # The seed is deterministic but varies for each dataset and each split of it
            max_evals=1000)
        best_score = np.min([t['result']['loss'] for t in trials.trials])
        if best_score < default:
            best = space_eval(param_grid, best)
        else:
            best = {}
    else:
        best = no_tune.copy()

    start = time.time()
    clf = clf_(**best)
    clf.fit(x, y)
    fit_time = time.time() - start
    start = time.time()
    if is_classification(metric_used):
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    inference_time = time.time() - start

    best = {'best': best}
    best['fit_time'] = fit_time
    best['inference_time'] = inference_time

    return pred, best
