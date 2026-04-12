# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Bharadwaj6903"
        repo_name = "mlops_mini_project"

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load latest model from registry (no staging)
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)

        if cls.new_model_version is None:
            raise ValueError("No model versions found in MLflow registry")

       # cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        #cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)



        client = mlflow.MlflowClient()
        # Get all versions
        versions = client.search_model_versions(f"name='{cls.new_model_name}'")

        if not versions:
            raise ValueError("No model versions found")

        # Get latest version
        latest_version = max(versions, key=lambda v: int(v.version))

        # Extract run_id
        run_id = latest_version.run_id

        if not run_id:
            raise ValueError("Run ID is missing for latest model version")
        
        print("Run ID:", run_id)
      

        # Correct artifact URI
        artifact_uri = f"runs:/{run_id}/model"

        print("Using artifact URI:", artifact_uri)

        # Load model
        cls.new_model = mlflow.pyfunc.load_model(artifact_uri)

        # Load vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()

        # Get all versions
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            return None

        # Pick latest version (max version number)
        latest_version = max(versions, key=lambda v: int(v.version))
        return latest_version.version

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Dummy input
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])]
        )

        # Prediction
        prediction = self.new_model.predict(input_df)

        # Input shape check
        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out())
        )

        # Output shape check
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        # Split features and labels
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Predict
        y_pred_new = self.new_model.predict(X_holdout)

        # Metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Thresholds
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assertions
        self.assertGreaterEqual(accuracy_new, expected_accuracy)
        self.assertGreaterEqual(precision_new, expected_precision)
        self.assertGreaterEqual(recall_new, expected_recall)
        self.assertGreaterEqual(f1_new, expected_f1)


if __name__ == "__main__":
    unittest.main()