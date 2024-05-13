from prefect import flow
import mlflow

from load.prep import prep_flow
from train.model_prep import model_prep_flow
from train.hpo import hpo_flow
from train.train import train_flow

HPO_EXPERIMENT_NAME = "project-MLops-hyperopt"
REG_EXPERIMENT_NAME = "project-MLops-best-models"

@flow
def main_flow():
    print("start main flow")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    prep_flow(data_path="./data/", dest_path="./models/")
    best_model = model_prep_flow("./models/")
    train_flow("./models/", best_model)
    

    #hpo_flow()

if __name__ == "__main__":
    main_flow()



# nicolaidegroot
# 6900d57ce8d4df31b0a445d890d726bf