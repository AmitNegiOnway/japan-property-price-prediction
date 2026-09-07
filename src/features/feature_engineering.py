import pandas as pd 
import numpy as np 
import os 
import yaml 

from sklearn.base import BaseEstimator , TransformerMixin 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,TargetEncoder
from sklearn.model_selection import train_test_split




def load_params(param_path:str)->dict:
    with open(param_path,'r') as file:
        params =yaml.safe_load(file)

    return  params   

params=load_params(param_path='param.yaml')
test_size=params['feature_engineering']['test_size']

# load df.csv
df=pd.read_csv('./data/raw/df.csv')

# split df 
X_train,X_test,Y_train,Y_test=train_test_split(df.drop(columns=['TradePrice']),df['TradePrice'],test_size=test_size,random_state=42)

# these features are not usefull 
drop_cols = ['Remarks','Renovation','No','UnitPrice','PricePerTsubo',
             'TotalFloorAreaIsGreaterFlag','FloorPlan','TimeToNearestStation',
             'MaxTimeToNearestStation','AreaIsGreaterFlag','Prefecture',
             'Municipality','Period','PrewarBuilding','FrontageIsGreaterFlag']



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




used_columns = [
    'Type', 'Region', 'LandShape', 'Purpose', 'Direction', 'CityPlanning',
    'NearestStation', 'DistrictName',
    'Structure', 'Use', 'Classification',
    
]

all_columns_after_drop = [col for col in X_train.columns if col not in drop_cols]

# Remaining columns that should be passed through unchanged
remaining_columns = [col for col in all_columns_after_drop if col not in used_columns]


# OHE , TargetEncoder       
column_transformer=ColumnTransformer([
    ('Only_OHE',OneHotEncoder(handle_unknown='ignore', sparse_output=False),
     ['Type' , 'Region' , 'LandShape', 'Purpose' , 'Direction' , 'CityPlanning']),
    ('NearestStation_TE',TargetEncoder(smooth=15, cv=5),['NearestStation']),
    ('DistrictName_TE',TargetEncoder(smooth=20, cv=7),['DistrictName']),
],remainder='drop')

# rare cat + OHE
custom_pipeline=Pipeline(steps=[
    ('rare_category',RareCategory(threshold=20)),
    ("OHE",OneHotEncoder(handle_unknown='ignore',sparse_output=False))  
])
rare_OHE_tranformer=ColumnTransformer([
    ('rare_ohe',custom_pipeline,['Structure', 'Use', 'Classification']),
],remainder='drop')



passthrough_transformer = ColumnTransformer([
    ('passthrough', 'passthrough', remaining_columns)
], remainder='drop')

# (Feature Union)
final_pipeline=FeatureUnion([
    ('column_transformer',column_transformer),
    ('rare_OHE_tranformer',rare_OHE_tranformer),
    ('Num_Property_Type',Num_Property_Type()),
    ('passthrough_transformer',passthrough_transformer)
])


full_pipeline = Pipeline([
    ('column_dropper', ColumnDropper(drop_cols)),
    ('feature_engineering', final_pipeline),
])


X_train_transformed = full_pipeline.fit_transform(X_train, Y_train)
X_test_transformed = full_pipeline.transform(X_test)
print(X_train_transformed.shape)
feature_names =full_pipeline.get_feature_names_out()  

X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
X_train_transformed_df.shape

X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)


import pickle 
pickle.dump(full_pipeline,open('models/nihon_pipeline','wb'))


data_path=os.path.join("./data/interim")
os.makedirs(data_path,exist_ok=True)


X_train_transformed_df.to_csv(os.path.join(data_path,'X_train_transformed_df.csv'),index=False)
X_test_transformed_df.to_csv(os.path.join(data_path,'X_test_transformed_df.csv'),index=False)
