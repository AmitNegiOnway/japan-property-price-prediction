import dagshub
import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/amitnegionway/japan-property-price-prediction.mlflow')
dagshub.init(repo_owner='amitnegionway', repo_name='mini-mlflow-temp', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)