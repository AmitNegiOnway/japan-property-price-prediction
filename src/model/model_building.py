# model building

import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import logging
import os
import contextlib


# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ----------------------------------------------------------------
# silence LightGBM C++ stderr warnings during fit
# ----------------------------------------------------------------
@contextlib.contextmanager
def quiet():
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _old_fd = os.dup(2)
    os.dup2(_devnull, 2)
    try:
        yield
    finally:
        os.dup2(_old_fd, 2)
        os.close(_devnull)
        os.close(_old_fd)


# ---------------------------------------------------------------
# # ----------------------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


# ----------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------
def prepare_data(df: pd.DataFrame, cat_features: list, drop_cols: list):
    """Drop cols, convert to pandas category, split, log-transform target."""
    try:
        from sklearn.model_selection import train_test_split

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

        y_train_log = np.log1p(y_train)
        y_test_log  = np.log1p(y_test)

        logger.debug('Data prepared: X_train %s, X_test %s', X_train.shape, X_test.shape)
        return X_train, X_test, y_train_log, y_test_log
    except Exception as e:
        logger.error('Error during data preparation: %s', e)
        raise


# ----------------------------------------------------------------
# training — params hardcoded here
# ----------------------------------------------------------------
def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray) -> lgb.LGBMRegressor:
    """Train the LightGBM model."""
    try:
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
            'objective':               'regression',
            'random_state':            42,
            'n_jobs':                  1,
            'verbosity':               -1,
        }

        model = lgb.LGBMRegressor(**params)

        with quiet():
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.log_evaluation(period=0)],
            )

        logger.debug('Model training completed')
        return model
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


# ----------------------------------------------------------------
# saving
# ----------------------------------------------------------------
def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------
def main():
    try:
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

        df = load_data('./data/raw/df.csv')
        X_train, X_test, y_train_log, y_test_log = prepare_data(
            df, cat_features, drop_cols
        )

        model = train_model(X_train, y_train_log, X_test, y_test_log)

        save_model(model, 'models/lgbm_model.pkl')

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()