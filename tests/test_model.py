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
        # DagsHub auth
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(
            "https://dagshub.com/Bharadwaj6903/mlops_mini_project.mlflow"
        )

        cls.new_model_name = "my_model"

        # 🔥 Get latest model version OBJECT
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{cls.new_model_name}'")

        if not versions:
            raise ValueError("No model versions found")

        latest_version = max(versions, key=lambda v: int(v.version))

        run_id = latest_version.run_id
        if not run_id:
            raise ValueError("Run ID missing")

        # ✅ IMPORTANT: correct artifact path
        artifact_uri = f"runs:/{run_id}/model"

        print("Run ID:", run_id)
        print("Artifact URI:", artifact_uri)

        # ✅ Load model (no registry call)
        cls.new_model = mlflow.pyfunc.load_model(artifact_uri)

        # Load vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])]
        )

        prediction = self.new_model.predict(input_df)

        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out())
        )

        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred_new = self.new_model.predict(X_holdout)

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        self.assertGreaterEqual(accuracy_new, 0.40)
        self.assertGreaterEqual(precision_new, 0.40)
        self.assertGreaterEqual(recall_new, 0.40)
        self.assertGreaterEqual(f1_new, 0.40)


if __name__ == "__main__":
    unittest.main()