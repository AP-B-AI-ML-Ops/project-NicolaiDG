# Describe your project

## Dataset

Ik ga de dataset gebruiken van 'Apple Quality'. Ik zou een API call doen naar de Kaggle URL om de data te extracten.

De link naar de dataset vind u hier:

https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality

## Project Explanation

Wat ik wil bereiken is een een simpele Machine Learning algoritme die de kwaliteit tussen Appels klassificieert tussen goed en slecht adhv verschillende eigenschappen van die appels.

De data die ik heb bevat bepaalde categorische eigenschappen van appels en ook een dimensie quality met good / bad. Het doel is moest iemand appels verwerken met bepaalde informatie van appels zoals de size , sweetness , crunchiness , Juiciness ,... dat de algoritme bepaald dat die appels goed zijn of niet.

Het is een vrij simpele algoritme, maar de kern erachter is om te focussen dat er een goeie pipeline wordt gebouwd dus daarom koos ik voor een simpele algoritme die een grote slagingskans heeft. Ik kan hierop hyperparameters zoeken en tuning erop uitvoeren om zo het model sterker en efficiënter te maken.

De dimensies in de dataset zijn:
* A_id
* Size
* Weight
* Sweetness
* Crunchiness
* Juiciness
* Ripeness
* Acidity
* Quality

## Flows & Actions

### Data inlezen

Ik zou een API request doen naar Kaggle om de data in te lezen in csv formaat zodat ik het kan gebruiken in mijn project. Dit gebeurt in de load file in de python file prep. Ik haal de file binnen als een zip bestand , pak die uit en zet die in de gewenste map.

### Preprosessing
Ik zou de data filteren op basis van de dimensies die ik nodig heb voor mijn project, de NULL values eruit filteren en bijvoorbeeld de data van 'Quality' in de dataset converteren naar numerieke waarde aangezien de data alleen uit goed of slecht bevat zodat ik hier makkelijk mee kan werken. Ik zou ook de datatypes controleren van de dimensies en eventueel naar de bijpassende datatype converteren. Ik zou ook de data splitsen in training en testset. Ik haal ook niet nuttige kolommen weg zodat het mij later geen problemen gaat geven. Dit gebeurt ook in de prep.py

### Model seargh
Ik zou een task maken die probeert meerdere modellen te testen en beste accuracy te vinden zodat die model gekozen kan worden om mijn project te maken. In model_prep.py bereken ik al op basis van de beste mean score welke klassificatie model het beste zou scoren voor de dataset. Die model zou ik in een variabele meenemen naar andere flow zodat het model daarop gebouwd kan worden.

### Model training
Trainen van een klassificatiemodel op basis van de best gekozen model van mijn model seargh task. Hier word de model geselecteerd met enkele random parameters en word die meegenomen naar de volgende flow. Dit speelt zich af in de train.py bestand.

### Hyperparameter tuning
De beste hyperparameters zoeken voor de geselecteerde klassificatiemodel. Hier word op basis van de best preseterende model voor de dataset de model verwerkt en de  hyperparameters berekent en naar mlflow gestuurd. Dit gebeurt in hpo.py

### Verwerking beste model
In register.py word de beste model gekozen op basis van de beste hyperparameters en opgeslagen als d ebeste model in MLFlow het word vervolgens ook opgeslagen in een map zodat het later opnieuw gebruikt kan worden.

### Evidently Rapport
In evidently rapport.py word ervoor gezorgd dat de beste model in een evidently rapport word gestoken om zo de prestaties van het model te gekijken. Vervolgens doen wij ongeveer hetzelfde in python file database_store.py waar wij de data eerst overzetten naar een postgreSQL om vervolgens in een Grafana dashboard de data weer te geven.


### Service implementatie
Tools die gebruikt werden zijn:

- MLflow
- Prefect
- Docker
- Grafana
- Adminer
- postgreSQL

### Applicatie handleiding

De eerste stap is te itereren naar .devcontainer -> 'docker-compose up -d --build' te typen in de terminal op alle instanties op te starten -> itereren terug naar de hoofdfolder van het project -> in de terminal 'mlflow ui --backend-store-uri sqlite:///mlflow.db' te verwerken -> de python file main.py te runnen of de bash script ./prefect te runnen en in prefect deployment de applicatie te runnen -> eenmaal aplicatie gerund is moet in grafana vanwege de datasource file niet werkte de dashboard nog zelf aagemaakt worden. In grafana moet in de settings connection een postgres verbinding aangemaakt worden met volgende

configuraties:
- Host URL = db:5432
- Database name = test
- Username : postgres
- Password : (zie .env)
- TLS/SSL mode : disable
- Postgres version : 15

Vervolgens kun je de database connectie testen of het werken. Hierna kan een dashboard hierop aangemaakt worden bij de dashboard tab. Bij add visualisation word de datasource die we gemaakt hebben gedrukt. Vanonder gaan we een code query maken. Deze query zou er moeten verwerkt worden: 'SELECT prediction_drift,num_drifted_columns,share_missing_values FROM metrics;'. Vervolgens word de query uitgevoerd en selecteer je de Gauge visualisation om de evidently ongeveer te reconstrueren. (in de map rapport in het project valt een html file van evidently ook te vinden)


### Versiebeheer:

- Docker version: 3.8
- postgres : 15
