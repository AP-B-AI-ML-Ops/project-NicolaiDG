import os
import pickle

import mlflow
import optuna
from optuna.samplers import TPESampler
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@task
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def optimize(X_train, y_train, X_val, y_val, num_trials, best_model):
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()

    def objective(trial):
        if isinstance(best_model, RandomForestClassifier):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 50, 1),
                "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, 1),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4, 1),
                "random_state": 42,
                "n_jobs": -1,
            }
            with mlflow.start_run():
                mlflow.log_params(params)
                model_instance = RandomForestClassifier(**params)
                model_instance.fit(X_train_copy, y_train_copy)  # Gebruik de kopieën
                y_pred = model_instance.predict(X_val)
                y_val_copy = y_val.copy()
                rmse = mean_squared_error(y_val_copy, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)
            return rmse

        if isinstance(best_model, KNeighborsClassifier):
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 10),
                "weights": trial.suggest_categorical(
                    "weights", ["uniform", "distance"]
                ),
                "algorithm": trial.suggest_categorical(
                    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
                ),
                "leaf_size": trial.suggest_int("leaf_size", 10, 50),
                "p": trial.suggest_int("p", 1, 2),
            }
            with mlflow.start_run():
                mlflow.log_params(params)
                model_instance = KNeighborsClassifier(**params)
                model_instance.fit(X_train_copy, y_train_copy)  # Gebruik de kopieën
                y_pred = model_instance.predict(X_val)
                y_val_copy = y_val.copy()
                rmse = mean_squared_error(y_val_copy, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)
            return rmse

        if isinstance(best_model, SVC):
            params = {
                "C": trial.suggest_loguniform("C", 0.1, 10.0),
                "kernel": trial.suggest_categorical(
                    "kernel", ["linear", "poly", "rbf", "sigmoid"]
                ),
                "degree": trial.suggest_int("degree", 1, 5),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "probability": True,
            }
            with mlflow.start_run():
                mlflow.log_params(params)
                model_instance = SVC(**params)
                model_instance.fit(X_train_copy, y_train_copy)  # Gebruik de kopieën
                y_pred = model_instance.predict(X_val)
                y_val_copy = y_val.copy()
                rmse = mean_squared_error(y_val_copy, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)
            return rmse

        if isinstance(best_model, LogisticRegression):
            params = {
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "C": trial.suggest_loguniform("C", 0.01, 10.0),
                "solver": trial.suggest_categorical(
                    "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
            }
            with mlflow.start_run():
                mlflow.log_params(params)
                model_instance = LogisticRegression(**params)
                model_instance.fit(X_train_copy, y_train_copy)  # Gebruik de kopieën
                y_pred = model_instance.predict(X_val)
                y_val_copy = y_val.copy()
                rmse = mean_squared_error(y_val_copy, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)
            return rmse

        if isinstance(best_model, DecisionTreeClassifier):
            params = {
                "max_depth": trial.suggest_int("max_depth", 1, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical(
                    "max_features", ["auto", "sqrt", "log2"]
                ),
            }
            with mlflow.start_run():
                mlflow.log_params(params)
                model_instance = DecisionTreeClassifier(**params)
                model_instance.fit(X_train_copy, y_train_copy)  # Gebruik de kopieën
                y_pred = model_instance.predict(X_val)
                y_val_copy = y_val.copy()
                rmse = mean_squared_error(y_val_copy, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)
            return rmse

        print("Ongeldige model")
        return float("inf")

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


@flow
def hpo_flow(model_path: str, num_trials: int, experiment_name: str, best_model):
    mlflow.set_experiment(experiment_name)

    mlflow.sklearn.autolog(disable=True)

    X_train, y_train = load_pickle(os.path.join(model_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(model_path, "val.pkl"))

    optimize(X_train, y_train, X_val, y_val, num_trials, best_model)
