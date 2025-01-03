import time
import math

import numpy as np

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials , space_eval, rand

from .saint_lib import SAINT

from tabular_prediction.utils import is_classification, preprocess_impute, eval_complete_f

param_grid = {
    'dim': hp.choice('dim', [32, 64, 128]), #,256
    'depth': hp.choice('depth', [2, 3, 6]), #,12
    'heads': hp.choice('heads', [2, 4, 8]),
    'dropout': hp.choice('dropout', [0, 0.2, 0.4, 0.6, 0.8]),
    'epochs': hp.choice('epochs', [2, 3]),
}

def eval_f(params, model_, x, y, metric_used, cv=None):
    model = model_(**params)
    _, val_loss_history = model.fit(x, y)
    return np.nanmin(val_loss_history)

def eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune,
                    cv=5, eval_f=eval_f, run_default=True):
    if not isinstance(max_time, list):
        max_time = [max_time]

    if no_tune is None:
        if run_default:
            default = eval_f({}, model_, x, y, metric_used, cv=cv)

        summary = {}
        trials = Trials()
        used_time = 0
        for i, stop_time in enumerate(max_time):
            time_budget = stop_time - used_time
            if time_budget <= 0:
                summary[stop_time] = {}
                summary[stop_time]['hparams'] = summary[max_time[i-1]]['hparams']
                summary[stop_time]['tune_time'] = summary[max_time[i-1]]['tune_time']
                continue

            start_time = time.time()
            def stop(trial, count=0):
                count += 1
                return (count + 1)/count * (time.time() - start_time) > time_budget, [count]

            best = fmin(
                fn=lambda params: eval_f(params, model_, x, y, metric_used, cv=cv),
                space={**param_grid, "directory": hp.choice('directory', [str(stop_time)])},
                algo=rand.suggest,
                #rstate=np.random.default_rng(int(y[:].sum() + stop_time) % 10000),
                early_stop_fn=stop,
                trials=trials,
                catch_eval_exceptions=True,
                verbose=True,
                # The seed is deterministic but varies for each dataset and each split of it
                max_evals=1000)
            best_loss = np.min([t['result']['loss'] for t in trials.trials])
            used_time += time.time() - start_time

            summary[stop_time] = {}
            if (not run_default) or (best_loss < default):
                summary[stop_time]['hparams'] = space_eval(param_grid, best)
            else:
                summary[stop_time]['hparams'] = {}
            summary[stop_time]['tune_time'] = used_time
    else:
        summary[stop_time]['hparams'] = no_tune.copy()

    for stop_time in summary:
        start = time.time()
        model = model_(**summary[stop_time]['hparams'])
        model.load_model(filename_extension="best", directory=str(stop_time))
        train_time = time.time() - start
        start = time.time()
        if is_classification(metric_used):
            pred = model.predict_proba(test_x)
        else:
            pred = model.predict(test_x)
        predict_time = time.time() - start

        summary[stop_time]['pred'] = pred
        summary[stop_time]['train_time'] = train_time
        summary[stop_time]['predict_time'] = predict_time

    return summary

def saint_predict(x, y, test_x, test_y, metric_used, cat_features=None, max_time=300, no_tune=None, run_id=""):
    x, y, test_x, test_y, cat_features = preprocess_impute(x, y, test_x, test_y,
        one_hot=False, impute=False, standardize=False, cat_features=cat_features)

    def model_(**params):
        return SAINT(n_features=x.shape[1], cat_features=cat_features,
            cat_dims=[len(np.unique(x[:, c])) for c in cat_features],
            is_classification=is_classification(metric_used),
            n_classes=len(np.unique(y)),
            run_id=run_id, **params)

    start_time = time.time()
    summary = eval_complete_f(x, y, test_x, model_, param_grid, metric_used, max_time, no_tune, eval_f=eval_f, run_default=False)
    end_time = time.time()
    return test_y, summary, end_time-start_time
