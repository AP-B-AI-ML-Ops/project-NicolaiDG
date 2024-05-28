import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from load.prep import start_ml_experiment
from prefect import flow
import mlflow


class TestStartMLExperiment(unittest.TestCase):
    @patch("mlflow.start_run")
    def test_start_ml_experiment_knn(self, mock_start_run):
        # Mock data
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y_train = pd.Series([0, 1, 0])
        model = KNeighborsClassifier()

        @flow
        def test_flow():
            start_ml_experiment(X_train, y_train, model)

        test_flow()

        # Ensure mlflow.start_run was called
        mock_start_run.assert_called_once()

    @patch("mlflow.start_run")
    def test_start_ml_experiment_svc(self, mock_start_run):
        # Mock data
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y_train = pd.Series([0, 1, 0])
        model = SVC()

        @flow
        def test_flow():
            start_ml_experiment(X_train, y_train, model)

        test_flow()

        # Ensure mlflow.start_run was called
        mock_start_run.assert_called_once()

    @patch("mlflow.start_run")
    def test_start_ml_experiment_rf(self, mock_start_run):
        # Mock data
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y_train = pd.Series([0, 1, 0])
        model = RandomForestClassifier()

        @flow
        def test_flow():
            start_ml_experiment(X_train, y_train, model)

        test_flow()

        # Ensure mlflow.start_run was called
        mock_start_run.assert_called_once()

    @patch("mlflow.start_run")
    def test_start_ml_experiment_lr(self, mock_start_run):
        # Mock data
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y_train = pd.Series([0, 1, 0])
        model = LogisticRegression()

        @flow
        def test_flow():
            start_ml_experiment(X_train, y_train, model)

        test_flow()

        # Ensure mlflow.start_run was called
        mock_start_run.assert_called_once()

    @patch("mlflow.start_run")
    def test_start_ml_experiment_dc(self, mock_start_run):
        # Mock data
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y_train = pd.Series([0, 1, 0])
        model = DecisionTreeClassifier()

        @flow
        def test_flow():
            start_ml_experiment(X_train, y_train, model)

        test_flow()

        # Ensure mlflow.start_run was called
        mock_start_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
