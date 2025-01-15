import os
import argparse
import torch
from multiprocessing import Pool
from tabular_prediction.methods import saint_predict
from tabular_prediction.metrics import accuracy_metric, balanced_accuracy_metric, cross_entropy_metric, auc_metric

# all datasets

from read_data import get_datasets
    

def run_evaluation(split, gpu_id=0, parallelize_datasets=False):
    max_time = [1, 5, 10, 30, 60, 120, 300, 600, 3600]

    data_dir, datasets = get_datasets(split)

    with open(f"../results/saint-classification-{split}.csv", "a") as f:
        f.write(','.join(["dataset", "acc", "bacc", "ce", "auc", "time"]))
        f.write('\n')
        f.flush()
        for i, dataset in enumerate(datasets):
            if parallelize_datasets:
                run_id="_".join([dataset.split(".")[0], str(split)])
            else:
                run_id = str(split)
                
            data = torch.load(os.path.join(data_dir, dataset), map_location='cpu')
            x_train, y_train, x_test, y_test = data["data"]
            cat_features = torch.where(data["cat_features"])[0]

            test_y, summary, _ = saint_predict(
                x_train, y_train, x_test, y_test, cat_features=cat_features, 
                metric_used=cross_entropy_metric, max_time=max_time, gpu_id=gpu_id, 
                run_id=run_id,
                )
            for stop_time in summary:
                pred = summary[stop_time]['pred']
                run_time = summary[stop_time]['tune_time'] + summary[stop_time]['train_time'] + summary[stop_time]['predict_time']
                f.write(','.join([dataset] + [f'{val:5.4f}' for val in [accuracy_metric(test_y, pred), balanced_accuracy_metric(test_y, pred), cross_entropy_metric(test_y, pred), auc_metric(test_y, pred), run_time]]))
                f.write('\n')
                f.flush()
                

parser = argparse.ArgumentParser()
parser.add_argument('split', type=int, help='split number - must be 1 to 6')
parser.add_argument('gpu', type=int, help='which gpu to use - 0, 1, ...')

args = parser.parse_args()

split = args.split
assert split in range(1,7)

print(f"starting split {split}")
run_evaluation(split=split, gpu_id=args.gpu)
print(f"completed split {split}")

