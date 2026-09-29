# register model

import json
import mlflow
import dagshub
import os 


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


def load_model_info(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    return model_info


# registration

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    model_uri = model_info['model_uri']
    
    # register the model
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

    # point the "staging" alias at this version
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias='staging',
        version=model_version.version,      # ← fixed: was model_version (the object)
    )

    print(f"Staging alias now points to version {model_version.version}")



# main



def main():
    model_info = load_model_info('reports/experiment_info.json')

    model_name = "my_model"
    register_model(model_name, model_info)


if __name__ == '__main__':
    main()






    