import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def is_classification(metric_used):
    if metric_used.__name__ in ["accuracy_metric", "cross_entropy_metric", "auc_metric", "balanced_accuracy_metric", "average_precision_metric"]:
        return True
    return False

def preprocess_impute(x, y, test_x, test_y, impute, one_hot, standardize, cat_features=[]):
    import warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn

    x, y, test_x, test_y = x.cpu().numpy(), y.cpu().long().numpy(), test_x.cpu().numpy(), test_y.cpu().long().numpy()

    if impute:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(x)
        x, test_x = imp_mean.transform(x), imp_mean.transform(test_x)

    if one_hot:
        def make_pd_from_np(x):
            data = pd.DataFrame(x)
            for c in cat_features:
                data.iloc[:, c] = data.iloc[:, c].astype('int')
            return data
        x, test_x = make_pd_from_np(x),  make_pd_from_np(test_x)
        transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_features)], remainder="passthrough")
        transformer.fit(x)
        x, test_x = transformer.transform(x), transformer.transform(test_x)

    if standardize:
        scaler = MinMaxScaler()
        scaler.fit(x)
        x, test_x = scaler.transform(x), scaler.transform(test_x)

    return x, y, test_x, test_y
