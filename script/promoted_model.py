# promote model — staging → champion

import os
import mlflow
import dagshub

# DagsHub / MLflow auth

dagshub_token=os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"]=dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"]=dagshub_token

dagshub_url="https://dagshub.com"
repo_owner="amitnegionway"
repo_name="japan-property-price-prediction"




# promotion

def promote_model(model_name: str = "my_model"):
    """Promote the current 'staging' model version to 'champion'."""

    client = mlflow.tracking.MlflowClient()

    # ---- get staging version ----
    staging_version = client.get_model_version_by_alias(
        model_name,
        alias="staging",
    ).version
    print(f"Staging version: {staging_version}")

    # ---- show current champion (if any) ----
    try:
        champ = client.get_model_version_by_alias(model_name, alias="champion")
        print(f"Current champion: version {champ.version}")
    except Exception:
        print("No champion model found yet.")

    # ---- promote staging → champion ----
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=staging_version,
    )

    print(f"Model version {staging_version} promoted to champion")


# main

if __name__ == "__main__":
    promote_model()