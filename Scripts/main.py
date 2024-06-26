
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from scipy import stats
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime



# Configuration de la page
st.set_page_config(page_title='Analyse de la Consommation Énergétique', layout="wide")
st.title('Analyse de la Consommation Énergétique')
st.set_option('deprecation.showPyplotGlobalUse', False)


model = load_model('my_model_0_01.keras')

def get_model():
    return model
model = get_model()

# Chargement des données
df_geo = pd.read_csv('donnée concatenées.csv', usecols=['annee', 'region', 'filiere', 'valeur'], encoding='Latin1', sep=';')
df = pd.read_csv('donnée concatenées.csv', encoding='Latin1', sep=';')
df_geo['annee'] = pd.to_datetime(df_geo['annee'], format='%Y')


# Définition des coordonnées pour les régions françaises
region_to_coords = {
    'Auvergne-Rhône-Alpes': [45.4473, 4.3859],
    'Bourgogne-Franche-Comté': [47.2805, 4.9994],
    'Bretagne': [48.2020, -2.9326],
    'Centre-Val de Loire': [47.7516, 1.6751],
    'Corse': [42.0396, 9.0129],
    'Grand Est': [48.6998, 6.1878],
    'Hauts-de-France': [50.4801, 2.7937],
    'Île-de-France': [48.8566, 2.3522],
    'Normandie': [49.1829, 0.3707],
    'Nouvelle-Aquitaine': [45.7074, 0.1532],
    'Occitanie': [43.8927, 3.2828],
    'Pays de la Loire': [47.7633, -0.3296],
    'Provence-Alpes-Côte dAzur': [43.9352, 6.0679]
}
df_geo['region'].fillna('Inconnu', inplace=True)
df_geo['filiere'] = df_geo['filiere'].astype(str)

# Onglets
tab1, tab2, tab3, tab4 = st.tabs(["Visualisations", "Géolocalisation","Statistiques","Predictions"])

with tab1:
    st.subheader("Visualisations diverses")
    # Sélection du type de graphique
    graph_type = st.selectbox("Choisir le type de graphique", 
                              ["Histogramme", "Boxplot" , "Heatmap"], key='graph_type')
    
    # Sélection des colonnes à visualiser
    selected_columns = st.multiselect("Sélectionnez les colonnes à visualiser", df.columns, key='columns')
    
    if st.button("Générer Graphique"):
        if not selected_columns:
            st.error("Veuillez sélectionner au moins une colonne.")
        else:
            # On convertit toutes les colonnes sélectionnées en numérique
            df_numeric = df[selected_columns].apply(pd.to_numeric, errors='coerce')
            df_numeric = df_numeric.dropna()

            if graph_type == "Histogramme":
                for column in selected_columns:
                    st.write(sns.histplot(df_numeric[column], kde=True))
                    st.pyplot()

            elif graph_type == "Boxplot":
                st.write(sns.boxplot(data=df_numeric))
                st.pyplot()

with tab2:
    st.subheader("Répartition de la consommation d’énergie par région")
    year_to_filter = st.slider('Année', int(df_geo['annee'].dt.year.min()), int(df_geo['annee'].dt.year.max()), key='year_slider')
    df_filtered = df_geo[df_geo['annee'].dt.year == year_to_filter]
    consumption_totals = df_filtered.groupby('region')['valeur'].sum().reset_index()
    consumption_totals['lat'] = consumption_totals['region'].apply(lambda x: region_to_coords.get(x, [None, None])[0])
    consumption_totals['lon'] = consumption_totals['region'].apply(lambda x: region_to_coords.get(x, [None, None])[1])
    consumption_totals['weight'] = consumption_totals['valeur'] / consumption_totals['valeur'].max()

    view_state = pdk.ViewState(latitude=46.2276, longitude=2.2137, zoom=5, pitch=50)
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=consumption_totals,
        get_position='[lon, lat]',
        get_weight="weight",
        opacity=0.9,
        threshold=0.5,
        aggregation='MEAN'
    )
    st.pydeck_chart(pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v9'
    ))

with tab3:
    st.subheader("Analyses statistiques")
    st.markdown("### Statistiques Descriptives")
    if st.button('Afficher les statistiques descriptives'):

        st.write(df.describe())

    st.markdown("### Corrélation entre les variables")
    if st.button('Afficher la matrice de corrélation'):
        st.write(df.corr())

################################################
# Préparation des données



import joblib
scaler_X = joblib.load('scaler_X.pkl')
scaler_y= joblib.load('scaler_y.pkl')
with tab4:
    st.subheader("Prédiction de la consommation énergétique")
    # Création du formulaire pour saisir les données de prédiction
    with st.form(key='my_form'):
        datetime_input = st.text_input(label="Entrez la date et l'heure (YYYY-MM-DD HH:MM:SS)", value="2024-01-01 12:00:00")
        submit_button = st.form_submit_button(label='Prédire')

    if submit_button:
        # Convertir la chaîne de date/heure en caractéristiques numériques
        date_time_obj = datetime.strptime(datetime_input, '%Y-%m-%d %H:%M:%S')
        hour = date_time_obj.hour
        day_of_week = date_time_obj.weekday()

        # Préparer les données pour le modèle
        input_features = np.array([[hour, day_of_week]])
        input_features = scaler_X.transform(input_features)  #
        
        # Faire la prédiction
        predicted_value_scaled = model.predict(input_features)  # Prédiction avec le modèle
        predicted_value = scaler_y.inverse_transform(predicted_value_scaled) 
        prediction = model.predict(input_features)
        predicted_value = prediction[0][0]

        st.write(f"La prédiction de la consommation énergétique pour {datetime_input} est {predicted_value}")
    
    