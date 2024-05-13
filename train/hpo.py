import os
import mlflow

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from prefect import task, flow

@task
def optimize(best_model, X, y):

    pipeline = Pipeline([('clf', best_model)])
    param_grid = {
        'clf__n_neighbors': [3, 5, 7],
        'clf__weights': ['uniform', 'distance']
    } 

    # Perform GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)


        # Print best parameters and best score
    print("\nBest Parameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)



@flow
def hpo_flow(best_model, X, y):
    optimize(best_model, X, y)  