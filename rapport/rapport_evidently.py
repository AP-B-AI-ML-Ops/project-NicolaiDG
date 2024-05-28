import os
import pickle

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from prefect import flow, task


@task
def open_pickle(pickle_file_path):
    with open(pickle_file_path, "rb") as file:
        model = pickle.load(file)
        return model


@task
def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


@task
def model_training(train_data, model, num_features):
    # Controleer de inhoud van de tuple
    if isinstance(train_data, tuple):
        # Veronderstel dat de eerste element de features bevat en de tweede element de targets
        X_train = train_data[0]
        y_train = train_data[1]
        # Controleer of de features en targets Pandas DataFrames/Series zijn
        if isinstance(X_train, pd.DataFrame) and isinstance(
            y_train, (pd.DataFrame, pd.Series)
        ):
            # Train het model met de geladen gegevens
            model.fit(X_train[num_features], y_train.values.ravel())
            train_preds = model.predict(X_train[num_features])
            train_data_with_preds = pd.concat(
                [X_train, y_train.rename("target")], axis=1
            )
            train_data_with_preds["prediction"] = train_preds
            return model, train_data_with_preds
        print("De geladen gegevens bevatten niet de verwachte types.")
    print("De geladen gegevens zijn geen tuple.")
    return None


@task
def validatie_data_predicties(val_data, model, num_features):
    # Controleer het type van de validatieset en de elementen daarin
    if isinstance(val_data, tuple):
        if (
            len(val_data) == 2
            and isinstance(val_data[0], pd.DataFrame)
            and isinstance(val_data[1], pd.Series)
        ):
            val_data = pd.concat([val_data[0], val_data[1].rename("target")], axis=1)
        else:
            print("De structuur van de validatieset-tuple is niet zoals verwacht.")
            return None
    else:
        print("De validatieset is geen tuple.")
        return None

    # Controleer nu of val_data een DataFrame is
    if isinstance(val_data, pd.DataFrame):
        # Veronderstel dat je categorical features hebt, voeg ze hier toe
        cat_features = []  # Voeg hier je categorical features toe als je die hebt
        all_features = num_features + cat_features

        # Voer de voorspellingen uit
        val_preds = model.predict(val_data[all_features])

        # Voeg de voorspellingen toe aan de validatieset
        val_data["prediction"] = val_preds
        return val_data
    print("De validatieset is geen Pandas DataFrame.")
    return None


@task
def evidently_html_rapport(num_features, train_data_with_preds, val_data):
    # Voor Evidently rapportage
    column_mapping = ColumnMapping(
        target="Quality",  # of None
        prediction="prediction",
        numerical_features=num_features,
    )

    # Welke metrics willen we gebruiken?
    report = Report(
        metrics=[
            ColumnDriftMetric(
                column_name="prediction"
            ),  # We willen prediction drift berekenen, omdat dit de belangrijkste metric in ons model is
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
    )

    # We moeten de referentie data, huidige data en kolom mapping doorgeven om een rapport te genereren
    report.run(
        reference_data=train_data_with_preds,
        current_data=val_data,
        column_mapping=column_mapping,
    )

    # Rapport opslaan
    report.save_html("rapport/evidently_report.html")


@flow
def rapport_flow(best_model_path, model_folder_path, val_path):
    # beste model ophalen
    model_locatie = best_model_path
    model = open_pickle(model_locatie)
    # model aanpassen zodat er gepredict kan worden
    num_features = [
        "Size",
        "Weight",
        "Sweetness",
        "Crunchiness",
        "Juiciness",
        "Ripeness",
        "Acidity",
    ]
    train_data = load_pickle(os.path.join(model_folder_path, "train.pkl"))
    result = model_training(train_data, model, num_features)
    if result is None:
        print("Model training failed.")
        return
    model, train_data_with_preds = result
    with open("models/model_with_pred.bin", "wb") as f_out:
        pickle.dump(model, f_out)
    # val ophalen
    val_data_path = (
        val_path  # Zorg ervoor dat je het juiste pad hebt voor je validatie data
    )
    val_data = load_pickle(val_data_path)
    # validatie predicties ophalen
    val_data = validatie_data_predicties(val_data, model, num_features)
    if val_data is None:
        print("Validation data prediction failed.")
        return
    val_data.to_parquet("data/reference.parquet")
    # evidently rapport in html maken
    evidently_html_rapport(num_features, train_data_with_preds, val_data)
