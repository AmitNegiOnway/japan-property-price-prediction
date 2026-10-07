import pickle
import pandas as pd
import numpy as np
import os 
import mlflow

# --------------------------------------------------------------------------------------------------------------------
# DagsHub / MLflow auth

dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")


os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


dagshub_url = "https://dagshub.com"
repo_owner = "amitnegionway"
repo_name = "japan-property-price-prediction"


# Set up MLflow tracking URI
mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)


# --------------------------------------------------------------------------------------------------------------------
# Load model from MLflow Model Registry

def get_champion_model_version(model_name):

    client = mlflow.MlflowClient()

    champion_model = client.get_model_version_by_alias(
        model_name,
        "champion"
    )

    return champion_model.version


model_name = "my_model"

model_version = get_champion_model_version(model_name)

model_uri = f"models:/{model_name}/{model_version}"

model = mlflow.pyfunc.load_model(model_uri)

# --------------------------------------------------------------------------------------------------------'


cat_features = [
    'Type', 'Region', 'DistrictName', 'NearestStation',
    'LandShape', 'Structure', 'Use', 'Purpose',
    'Direction', 'Classification', 'CityPlanning'
]


# Load training data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "df.csv"))

drop_cols = [
            'Remarks', 'Renovation', 'No', 'UnitPrice', 'PricePerTsubo',
            'TotalFloorAreaIsGreaterFlag', 'FloorPlan', 'TimeToNearestStation',
            'MaxTimeToNearestStation', 'AreaIsGreaterFlag', 'Prefecture',
            'Municipality', 'Period', 'PrewarBuilding', 'FrontageIsGreaterFlag','TradePrice',
        ]


df=df.drop(columns=drop_cols)

# Create category map from training data
category_map = {}

for col in cat_features:
    df[col] = df[col].astype('category') 
    category_map[col] = df[col].cat.categories.tolist()


# Load trained model



class Predict:

    def __init__(self, user_input):
        self.user_input = user_input

    def house_price_prediction(self):

        data = pd.DataFrame([self.user_input.model_dump()])

        # Convert categorical columns using training categories
        for col in cat_features:
            data[col] = pd.Categorical(
                data[col],
                categories=category_map[col]
            )

        # Predict and convert log prediction back to original price
        result = np.expm1(model.predict(data))

        return result[0]