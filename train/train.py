import os
import pickle
import mlflow
from prefect import task, flow


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

@task
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def start_ml_experiment(X_train, y_train, model):
    with mlflow.start_run():
        model.fit(X_train, y_train)

@task 
def model_zoeker(best_model):
    
    classifiers = {
        'Logistic Regression': LogisticRegression(solver='lbfgs', C=1.0),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, criterion='gini'),
        'Random Forest': RandomForestClassifier(n_estimators=100, criterion='gini'),
        'SVM': SVC(kernel='linear', C=1.0),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance')  # 'uniform' of 'distance'
    }


    for model_name, model_instance in classifiers.items():
        if isinstance(best_model, type(model_instance)):
            return model_instance

@flow
def train_flow(model_path: str, best_model):
    mlflow.set_experiment("MLops-project-train")
    mlflow.sklearn.autolog()
    
    X_train, y_train = load_pickle(os.path.join(model_path, "train.pkl"))
    model = model_zoeker(best_model)

    start_ml_experiment(X_train, y_train, model)
