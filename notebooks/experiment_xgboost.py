# Import necessary libraries
import mlflow
import mlflow.sklearn
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator , TransformerMixin 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,TargetEncoder
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score,
    mean_absolute_percentage_error
)
import os
from sklearn.metrics import r2_score


import dagshub 
mlflow.set_tracking_uri('https://dagshub.com/amitnegionway/japan-property-price-prediction.mlflow')
dagshub.init(repo_owner='amitnegionway',repo_name='japan-property-price-prediction',mlflow=True)


df=pd.read_csv('./data/raw/df.csv')
X_train,X_test,y_train,y_test=train_test_split(df.drop(columns=['TradePrice']),df['TradePrice'],test_size=0.2,random_state=42)


from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        self._fitted_ = True   
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        # Return the remaining column names
        if input_features is None:
            return None
        return [f for f in input_features if f not in self.columns_to_drop]


# Custom Rare Category estimator
class RareCategory(BaseEstimator,TransformerMixin):
    def __init__(self,threshold,name='other'):
        self.threshold=threshold
        self.name=name
        self.rare_category=None
        
    def fit (self,X,y=None):
        self._fitted_ = True   
        count=X.value_counts()
        self.rare_category=count[count.values<self.threshold].index
        return self

    def transform(self,X):
        return X.where(~X.isin(self.rare_category),self.name)

    def get_feature_names_out(self, input_features=None):
        return input_features

class Num_Property_Type(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        self._fitted_ = True   
        return self 
        
    def transform(self,X):
        X=X.copy()
        flags = pd.DataFrame({
            'has_building': (~X['Type'].isin(['Agricultural Land', 'Forest Land', 'Residential Land(Land Only)'])).astype(int),
            'is_agricultural': (X['Type'] == 'Agricultural Land').astype(int),
            'is_land_only': X['Type'].isin(['Agricultural Land', 'Forest Land', 'Residential Land(Land Only)']).astype(int)
        })
        return flags.values   

    def get_feature_names_out(self, input_features=None):
        return ['has_building', 'is_agricultural', 'is_land_only']

import pickle

with open('models/nihon_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

y_test_log  = np.log1p(y_test)
y_train_log = np.log1p(y_train)
X_train_tranform=pipeline.fit_transform(X_train,y_train_log)
X_test_tranform=pipeline.transform(X_test)


# Start the parent run for hyperparameter tuning
mlflow.set_experiment("experiment")
with mlflow.start_run():
    params = {
    'n_estimators': 2100,
    'max_depth': 9,
    'learning_rate': 0.011386743547457483,
    'subsample': 0.7637064598239472,
    'colsample_bytree': 0.6437822103924169,
    'colsample_bylevel': 0.773148298066986,
    'gamma': 0.627396002672545,
    'reg_lambda': 0.17231780498389035,
    'reg_alpha': 0.003562459834801784,
    'min_child_weight': 5,
    'random_state': 42,
    'tree_method': 'hist',
    'objective': 'reg:squarederror',
}
    mlflow.log_params(params)
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train_tranform, y_train_log,
        eval_set=[(X_test_tranform, np.log1p(y_test))],
        verbose=False,
    )
    y_train_pred_log = model.predict(X_train_tranform)
    y_test_pred_log  = model.predict(X_test_tranform)

    # -----------------------------------------------------------
    # Regression metrics (in log space)
    # -----------------------------------------------------------
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

    for k, v in train_metrics.items():
        mlflow.log_metric(f"train_{k}", v)
    for k, v in test_metrics.items():
        mlflow.log_metric(f"test_{k}", v)

    mlflow.xgboost.log_model(model, name='xgb_model')
    mlflow.log_artifact('models/nihon_pipeline.pkl')
    mlflow.log_artifact(__file__)        


    
