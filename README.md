# TabularPrediction
A repository hosting popular tabular prediction methods. Tuning is automatically performed for each method.


## Installation

### 1. Create Conda Environment
```bash
conda create -n tab_env python=3.8
conda activate tab_env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```


## Run

Prediction on custom datasets can be performed with the following function call. See [classification/](classification/) for examples.
```python
*method*_predict(
  x_train, y_train, x_test, y_test,
  cat_features=cat_features,
  metric_used=metric_used,
  max_time=max_time
)
```

`x_train`, `y_train`, `x_test`, `y_test` are the data (loaded as torch tensors); `cat_features` is a list of indices indicating which columns of `x` are categorical; `metric_used` is the evaluation function to be used as the objective; `max_time` is the time budget (in seconds).
