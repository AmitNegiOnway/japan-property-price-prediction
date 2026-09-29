# model evaluation — LightGBM regression

import numpy as np
import pandas as pd
import pickle
import json
import os

import mlflow
import mlflow.lightgbm
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score,
    mean_absolute_percentage_error,
)

# DagsHub / MLflow auth

dagshub_token=os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"]=dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"]=dagshub_token

dagshub_url="https://dagshub.com"
repo_owner="amitnegionway"
repo_name="japan-property-price-prediction"


# loaders

def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, low_memory=False)


# ================================================================
# preprocessing — same logic as model_building.py
# ================================================================
def prepare_test_data(df: pd.DataFrame):
    """Reproduce the exact split + preprocessing from model_building."""
    cat_features = [
        'Type', 'Region', 'DistrictName', 'NearestStation', 'LandShape',
        'Structure', 'Use', 'Purpose', 'Direction', 'Classification', 'CityPlanning',
    ]

    drop_cols = [
        'Remarks', 'Renovation', 'No', 'UnitPrice', 'PricePerTsubo',
        'TotalFloorAreaIsGreaterFlag', 'FloorPlan', 'TimeToNearestStation',
        'MaxTimeToNearestStation', 'AreaIsGreaterFlag', 'Prefecture',
        'Municipality', 'Period', 'PrewarBuilding', 'FrontageIsGreaterFlag',
    ]

    df_light = df.drop(columns=drop_cols)

    for col in cat_features:
        df_light[col] = df_light[col].astype('category')
        if '__MISSING__' not in df_light[col].cat.categories:
            df_light[col] = df_light[col].cat.add_categories(['__MISSING__'])
        df_light[col] = df_light[col].fillna('__MISSING__')

    X_train, X_test, y_train, y_test = train_test_split(
        df_light.drop(columns=['TradePrice']),
        df_light['TradePrice'],
        test_size=0.2,
        random_state=42,
    )

    y_test_log = np.log1p(y_test)
    return X_test, y_test_log


# ================================================================
# evaluation (regression metrics)
# ================================================================
def evaluate_model(model, X_test, y_test_log: np.ndarray) -> dict:
    y_pred_log = model.predict(X_test)
    mse = mean_squared_error(y_test_log, y_pred_log)

    metrics_dict = {
        'R2':   r2_score(y_test_log, y_pred_log),
        'EVS':  explained_variance_score(y_test_log, y_pred_log),
        'MAE':  mean_absolute_error(y_test_log, y_pred_log),
        'MSE':  mse,
        'RMSE': np.sqrt(mse),
        'MAPE': mean_absolute_percentage_error(y_test_log, y_pred_log),
    }

    # raw-space (¥) metrics for business reporting
    y_true_raw = np.expm1(y_test_log)
    y_pred_raw = np.expm1(y_pred_log)

    metrics_dict['MAE_yen']  = mean_absolute_error(y_true_raw, y_pred_raw)
    metrics_dict['RMSE_yen'] = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    metrics_dict['MAPE_yen'] = mean_absolute_percentage_error(y_true_raw, y_pred_raw)
    metrics_dict['R2_yen']   = r2_score(y_true_raw, y_pred_raw)

    return metrics_dict


# ================================================================
# savers
# ================================================================
def save_metrics(metrics: dict, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)


# ================================================================
# main
# ================================================================
def main():
    mlflow.set_experiment("dvc-pipeline")

    with mlflow.start_run() as run:
        # ---- load model ----
        model = load_model('./models/lgbm_model.pkl')

        # ---- load raw data + reproduce the split ----
        df = load_data('./data/raw/df.csv')
        X_test, y_test_log = prepare_test_data(df)

        # ---- evaluate ----
        metrics = evaluate_model(model, X_test, y_test_log)

        # ---- save metrics locally ----
        save_metrics(metrics, 'reports/metrics.json')

        # ---- log metrics to MLflow ----
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # ---- log params (skip non-serializable values) ----
        if hasattr(model, 'get_params'):
            for param_name, param_value in model.get_params().items():
                if isinstance(param_value, (int, float, str, bool)):
                    mlflow.log_param(param_name, param_value)


                # ---- log model ----
        logged_model = mlflow.lightgbm.log_model(model, name="model")

        # ---- save model_uri for the next stage ----
        os.makedirs('reports', exist_ok=True)
        with open('reports/experiment_info.json', 'w') as f:
            json.dump({"model_uri": logged_model.model_uri}, f, indent=4)

                

        # ---- log artifact files ----
        mlflow.log_artifact('reports/metrics.json')
        mlflow.log_artifact('reports/experiment_info.json')

        print(f"Metrics: {metrics}")
        print(f"Model URI: {logged_model.model_uri}")


if __name__ == '__main__':
    main()