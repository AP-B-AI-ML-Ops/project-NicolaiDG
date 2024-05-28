import os
import subprocess

import pandas as pd
from prefect import flow, task
from sklearn.preprocessing import LabelEncoder


@task
def download_and_unzip_dataset(dataset_url, destination_dir):
    command = f"kaggle datasets download -d {dataset_url} -p {destination_dir} --unzip"
    with subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        _, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error: {stderr.decode()}")
        else:
            print(f"Dataset downloaded and extracted successfully to {destination_dir}")

            # Find the downloaded ZIP file
            zip_file = next(
                (f for f in os.listdir(destination_dir) if f.endswith(".zip")), None
            )
            if zip_file:
                zip_file_path = os.path.join(destination_dir, zip_file)
                os.remove(zip_file_path)
                print(f"ZIP file {zip_file_path} has been deleted")


@task
def read_dataset_to_csv(dataset):
    ds = pd.read_csv(f"./data/{dataset}")
    return ds


@task
def preprossessing(dataset, column_name, target):
    # null waardes verwijderen in rijen
    dataset = dataset.dropna()
    # de niet nuttige kolommen / data weghalen
    dataset = dataset[dataset["Quality"] != "Created_by_Nidula_Elgiriyewithana"]
    dataset = dataset[dataset["Acidity"] != "Created_by_Nidula_Elgiriyewithana"]
    dataset = dataset.drop(column_name, axis=1)
    # numerieke encoding van doel variabele
    label_encoder = LabelEncoder()
    dataset[target] = label_encoder.fit_transform(dataset[target])

    # Opslaan van de dataset met wijzigingen
    save_path = os.path.join("data", "processed_dataset.csv")
    dataset.to_csv(save_path, index=False)


@flow
def prep_flow():
    dataset_url = "nelgiriyewithana/apple-quality"
    destination_dir = "./data"
    download_and_unzip_dataset(dataset_url, destination_dir)

    dataset = read_dataset_to_csv("apple_quality.csv")
    preprossessing(dataset, "A_id", "Quality")
