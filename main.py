import matplotlib
import mlflow
from prefect import flow

from database.database_store import database_store_flow
from load.model_prep import model_prep_flow
from load.prep import prep_flow
from rapport.rapport_evidently import rapport_flow
from train.hpo import hpo_flow
from train.register import register_flow
from train.train import train_flow

matplotlib.use(
    "agg"
)  # Gebruik Agg-backend om interactie met Tkinter te vermijden dit geeft ERRORS anders


HPO_EXPERIMENT_NAME = "project-MLops-hyperopt"
REG_EXPERIMENT_NAME = "project-MLops-best-models"


@flow
def main_flow():

    print("start main flow")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    prep_flow()
    best_model = model_prep_flow("./models/")
    train_flow("./models/", best_model)
    hpo_flow("./models/", 5, HPO_EXPERIMENT_NAME, best_model)
    register_flow("./models/", 5, REG_EXPERIMENT_NAME, HPO_EXPERIMENT_NAME, best_model)

    rapport_flow("./models/best_model/model.pkl", "./models/", "./models/val.pkl")
    database_store_flow()


if __name__ == "__main__":
    main_flow()

# mlflow ui --backend-store-uri sqlite:///mlflow.db

# pylint --recursive=y .
# black --diff .
# black .
# isort --diff
# pre-commit sample-config | out-file .pre-commit-config.yaml -encoding utf8
