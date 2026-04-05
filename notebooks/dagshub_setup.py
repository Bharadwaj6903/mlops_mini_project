import dagshub

import mlflow
dagshub.init(repo_owner='Bharadwaj6903', repo_name='mlops_mini_project', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)