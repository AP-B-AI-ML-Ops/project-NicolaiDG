import os
import pandas as pd
import opendatasets as od
from prefect import task, flow
from sklearn.preprocessing import LabelEncoder


@task
def data_download(url):
    dataset_url = url
    od.download(dataset_url)
    download_dir = './apple-quality'
    destination_dir = './data'

    # Verplaats het CSV-bestand naar de gewenste locatie
    os.makedirs(destination_dir, exist_ok=True)  # Zorg ervoor dat de bestemmingsmap bestaat
    os.rename(os.path.join(download_dir, 'apple_quality.csv'), os.path.join(destination_dir, 'apple_quality.csv'))
    os.rmdir(download_dir)

@task
def read_dataset_to_csv(dataset):
    ds = pd.read_csv(f'./data/{dataset}')
    return ds

@task
def preprossessing(dataset,column_name, target):
    # null waardes verwijderen in rijen
    dataset = dataset.dropna()
    # de niet nuttige kolommen / data weghalen
    dataset = dataset[dataset['Quality'] != 'Created_by_Nidula_Elgiriyewithana']
    dataset = dataset[dataset['Acidity'] != 'Created_by_Nidula_Elgiriyewithana']
    dataset=dataset.drop(column_name,axis=1)
    # numerieke encoding van doel variabele
    label_encoder = LabelEncoder()
    dataset[target] = label_encoder.fit_transform(dataset[target])

    # Opslaan van de dataset met wijzigingen
    save_path = os.path.join('data', 'processed_dataset.csv')
    dataset.to_csv(save_path, index=False)

@flow
def prep_flow(data_path: str , dest_path: str):

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(dest_path, exist_ok=True)

    data_download(url = "https://www.kaggle.com/nelgiriyewithana/apple-quality")
    dataset = read_dataset_to_csv("apple_quality.csv")
    preprossessing(dataset, 'A_id', 'Quality')
