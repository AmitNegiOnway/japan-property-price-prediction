import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import contextlib

import mlflow
import mlflow.lightgbm
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score,
    mean_absolute_percentage_error,
)

# ================================================================
# 1. Silence LightGBM C++ stderr warnings
# ================================================================
@contextlib.contextmanager
def quiet():
    """Redirect stderr to /dev/null during the block."""
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _old_fd  = os.dup(2)
    os.dup2(_devnull, 2)
    try:
        yield
    finally:
        os.dup2(_old_fd, 2)
        os.close(_devnull)
        os.close(_old_fd)

# ================================================================
# 2. MLflow / DagsHub setup
# ================================================================
mlflow.set_tracking_uri(
    'https://dagshub.com/amitnegionway/japan-property-price-prediction.mlflow'
)
dagshub.init(
    repo_owner='amitnegionway',
    repo_name='japan-property-price-prediction',
    mlflow=True,
)
mlflow.set_experiment('LightGBM hyperparameter tuning')   # your exact name, unchanged

# ================================================================
# 3. Load data + split
# ================================================================
df = pd.read_csv('./data/raw/df.csv', low_memory=False)

df_light = df.drop(columns=[
    'Remarks', 'Renovation', 'No', 'UnitPrice', 'PricePerTsubo',
    'TotalFloorAreaIsGreaterFlag', 'FloorPlan', 'TimeToNearestStation',
    'MaxTimeToNearestStation', 'AreaIsGreaterFlag', 'Prefecture',
    'Municipality', 'Period', 'PrewarBuilding', 'FrontageIsGreaterFlag',
])

cat_features = [
    'Type', 'Region', 'DistrictName', 'NearestStation', 'LandShape',
    'Structure', 'Use', 'Purpose', 'Direction', 'Classification', 'CityPlanning',
]

# Convert to pandas category + handle missing
for col in cat_features:
    df_light[col] = df_light[col].astype('category')
    if '__MISSING__' not in df_light[col].cat.categories:
        df_light[col] = df_light[col].cat.add_categories(['__MISSING__'])
    df_light[col] = df_light[col].fillna('__MISSING__')

X_train, X_test, y_train, y_test = train_test_split(
    df_light.drop(columns=['TradePrice']),
    df_light['TradePrice'],                # ← fixed: use df_light, not df
    test_size=0.2,
    random_state=42,
)

y_train_log = np.log1p(y_train)
y_test_log  = np.log1p(y_test)

# ================================================================
# 4. Train + evaluate + log to MLflow
# ================================================================
with mlflow.start_run(run_name="lgbm_optuna_best"):

    # ---- your exact params, unchanged ----
    params = {
        'n_estimators':            1600,
        'learning_rate':           0.03152942978058161,
        'max_depth':               12,
        'num_leaves':              150,
        'feature_fraction':        0.5003829762218786,
        'min_child_samples':       46,
        'min_split_gain':          0.01107755282149725,
        'min_sum_hessian_in_leaf': 0.010577606465395677,
        'reg_lambda':              0.10194612038857746,
        'reg_alpha':               0.004357939922467093,
        'max_bin':                 68,
        # ---- static keys added (needed for reproducibility) ----
        'objective':               'regression',
        'random_state':            42,
        'n_jobs':                  1,
        'verbosity':               -1,
    }

    model = lgb.LGBMRegressor(**params)

    # ---- silence C++ stderr warnings during fit ----
    with quiet():
        model.fit(
            X_train, y_train_log,
            eval_set=[(X_test, y_test_log)],
            callbacks=[lgb.log_evaluation(period=0)],
        )

    # ---- predictions ----
    y_train_pred_log = model.predict(X_train)
    y_test_pred_log  = model.predict(X_test)

    # ---- metrics ----
    def reg_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            'R2':   r2_score(y_true, y_pred),
            'EVS':  explained_variance_score(y_true, y_pred),
            'MAE':  mean_absolute_error(y_true, y_pred),
            'MSE':  mse,
            'RMSE': np.sqrt(mse),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        }

    train_metrics = reg_metrics(y_train_log, y_train_pred_log)
    test_metrics  = reg_metrics(y_test_log,  y_test_pred_log)

    # ---- log params + metrics ----
    mlflow.log_params(params)

    for k, v in train_metrics.items():
        mlflow.log_metric(f"train_{k}", v)
    for k, v in test_metrics.items():
        mlflow.log_metric(f"test_{k}", v)

    # ---- log model (LightGBM flavor, correct name) ----
    mlflow.lightgbm.log_model(model, name='lightGBM_model')
    mlflow.log_artifact(__file__)

    # ---- verification ----
    print("=" * 50)
    print(f"{'Metric':<10}{'Train':>14}{'Test':>14}")
    print("=" * 50)
    for k in train_metrics:
        print(f"{k:<10}{train_metrics[k]:>14.4f}{test_metrics[k]:>14.4f}")
    print("=" * 50)