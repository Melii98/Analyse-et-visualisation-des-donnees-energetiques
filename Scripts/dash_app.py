import streamlit as st
from Data_Analysis import *
import pandas as pd
import matplotlib.pyplot as plt



file_path = 'eco2mix-national-tr.csv'
# Chargement des données
df = load_data(file_path)
"""Menu latéral pour choisir le type de graphique à afficher"""
option = st.sidebar.selectbox(
    'Choisissez une analyse à afficher:',
    ('Consommation par source', 'Consommation au fil du temps', 'Relation énergie-CO2', 'Matrice de corrélation', 'Prévision ARIMA', 'Boxplot de la consommation', 'Consommation par année', 'Violon de la consommation')
)
# Afficher le graphique choisi
if option == 'Consommation par source':
    st.title("Consommation d'énergie par source")
    fig = plot_energy_consumption_by_source(df)
    st.pyplot(fig)
elif option == 'Consommation au fil du temps':
    st.title("Évolution de la consommation d'énergie au fil du temps")
    fig = plot_energy_consumption_over_time(df)
    st.pyplot(fig)
