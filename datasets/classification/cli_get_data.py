import os
import func_timeout
assert os.getenv("OpenML_API_KEY") or os.getenv("OPENML_API_KEY"), "OpenML_API_KEY or OPENML_API_KEY needs to be defined in order for openML API to work!"

import math
import openml
import numpy as np

import torch
import requests

openml.config.apikey = os.getenv("OpenML_API_KEY", default=os.getenv("OPENML_API_KEY"))

def direct_download(url):
    filename = url.split("/")[-1]    

    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully.")
    else:
        print("Error downloading file:", response.status_code)


def download_openml_suite(suite_id=99, max_features=500, shuffle=True,
                          split_min=0.88, split_max=0.92, seed=None):
    if seed is not None: 
        np.random.seed(seed)
        split_digit = str(seed)[-1]
        split_dir = f"OpenML-CC18-{split_digit}"
    else:
        split_dir = "OpenML-CC18"
        
    os.makedirs(split_dir, exist_ok=True)

    benchmark_suite = openml.study.get_suite(suite_id=suite_id)
    datalist = openml.datasets.list_datasets(data_id=benchmark_suite.data, output_format='dataframe')

    n_classes = []
    for _, ds in enumerate(datalist.index):
        entry = datalist.loc[ds]
        name = entry['name']
 
        if os.path.exists(f"{name}.pt"):
            continue
        did = entry['did']
        print('Downloading', name, did, '..')
        
        # get the raw arff files:
        data_file_location = os.path.abspath(os.path.join(".", dataset.url.split("/")[-1]))
        
        # skip if the arff file is already present:
        if not os.path.exists(data_file_location):
            dataset = openml.datasets.get_dataset(int(did), download_qualities=True, download_features_meta_data=True)
        
        # el hack! use a local arff file, instead of the hosted arff at the url:
        dataset.data_file = data_file_location
        dataset.parquet_file = None

        # print(dataset)
        if not os.path.exists(dataset.data_file):
            print(f"Whoops! canna find the local arff file at {dataset.data_file}." 
                  f"Will attempt to load it directly from {dataset.url}"
            )
            direct_download(dataset.url)
            assert os.path.exists(dataset.data_file)

        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        if X is None: continue

        # cat_columns = X.select_dtypes(['category', 'object']).columns
        # for col in cat_columns:
        #     try:
        #         X[col].astype(np.float32)
        #     except:
        #         X[col] = X[col].astype('category').cat.codes
        for i, col in enumerate(X.columns):
            if not categorical_indicator[i]:
                try:
                    X[col].astype(np.float32)
                    continue
                except:
                    categorical_indicator[i] = True
            X[col] = X[col].astype('category').cat.codes

        X = X.values.astype('float32')

        N, F = X.shape
        #if F > max_features: continue

        n_classes.append(y.astype('category').cat.categories.size)
        y = y.astype('category').cat.codes.values

        if shuffle:
            perm = np.random.permutation(N)
            X = X[perm, :]
            y = y[perm]

        test_size = N - int(N*np.random.uniform(split_min, split_max))
        test_size = min(test_size, 1000)

        X_train, X_test = X[:(-test_size), :], X[(-test_size):, :]
        y_train, y_test = y[:(-test_size)], y[(-test_size):]
        X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))
        dataset = {
            "data": (X_train, y_train, X_test, y_test),
            "cat_features": torch.tensor(categorical_indicator, dtype=torch.long),
            "attribute_names": attribute_names
        }
        torch.save(dataset, os.path.join(split_dir, f'{name}.pt'))

    return n_classes


def system_adaptable_download_openml_suite(seed=41):
    print("Let the data download, begin!")
    try:
        # time-limited version of download_openml_suite(seed=seed):
        max_wait_time = 10 # seconds
        my_square = func_timeout.func_timeout(max_wait_time, download_openml_suite, args=[seed])
        # except func_timeout.FunctionTimedOut:
    except Exception as e:  
        print("Attempting to use proxy black magic to download data!")
        import envfuncs
        with envfuncs.proxy_context(select="authproxy"):
            download_openml_suite(seed=seed)


if __name__ == "__main__":
    for seed in range(41, 47):
        system_adaptable_download_openml_suite(seed)