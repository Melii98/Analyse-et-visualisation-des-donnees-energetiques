# **Analyse et Visualisation des Données Énergétiques**

## **Description**

Ce projet consiste en une série de scripts Python pour l'analyse et la visualisation des données énergétiques de la France. Il utilise des données publiques pour examiner la consommation énergétique, la production par différentes sources, et les émissions de CO2. Le projet inclut des étapes de prétraitement des données, une analyse exploratoire, et la visualisation interactive des tendances énergétiques.

## **Installation et Configuration**

Pour exécuter ce projet, vous aurez besoin de Python 3.8+ ainsi que des paquets suivants :

- Pandas
- NumPy
- Matplotlib
- Seabourn
- Dash
- Streamlit

Vous pouvez installer toutes les dépendances nécessaires en exécutant :
'pip install -r requirements.txt' 

## **Utilisation**

Pour lancer l'analyse des données, exécutez chaque script dans l'ordre suivant :

- `Preprocessing.py` pour nettoyer et préparer les données à être exploités.
- `energy_analysis_visulation.py` pour générer des visualisations statiques des données analysées via Streamlit, une explication de chaque visualtion est disponible en selectionnant la visualisation que l'on souhaite observer sur le menu deroulant à gauche.
- `geolocalisation.py` permet de générer une heatmap montrant la consommation énergetique par regionsen France 
- `main.py` pour démarrer une interface Streamlit qui permet une exploration interactive des données (visualisation statistique avec possibilité  de choisir le type de visualisation et géolocalisation et analyses statistiques )
- `pred.py`script permettant de faire une prediction de la consommation d'énérgie 


## **Fonctionnalités**

- **Nettoyage des Données** : Suppression des valeurs manquantes et des colonnes non pertinentes.
- **Analyse Exploratoire** : Calcul des statistiques descriptives et analyse des tendances.
- **Visualisation des Données** : Création de graphiques statiques et interactifs pour visualiser les tendances de la production et de la consommation énergétique.
- **Application  Interactive** : Une interface streamlit pour explorer les données de manière interactive.
- **Predicitons** : construction d'un modele de prediction de la consommation 


