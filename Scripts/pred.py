
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np 
import matplotlib.pyplot as plt
import joblib


# Chemin vers le fichier CSV
chemin_fichier = r'donnée concatenées - DL.csv'

# Charger les données
data = pd.read_csv(chemin_fichier, delimiter=';', encoding='latin1')

# Définition de la fonction pour supprimer les outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Suppression des outliers pour chaque colonne numérique
numerical_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
for col in numerical_cols:
    data = remove_outliers(data, col)
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(inplace=True)

    
# Remplacer les valeurs non numériques par NaN et imputer les valeurs manquantes
data['consommation'] = pd.to_numeric(data['consommation'], errors='coerce')
data.fillna(data.mean(), inplace=True)  # Utilisation d'imputation pour toutes les colonnes

# Préparation des données
scaler_X = MinMaxScaler()
scaler_y = StandardScaler()


X = data[numerical_cols]
X_scaled = scaler_X.fit_transform(X)
y = data['consommation']
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train_scaled = scaler_X.fit_transform(X_train)  # Ajustement et transformation des données d'entraînement

# Sauvegarde du scaler pour une utilisation ultérieure lors de la prédiction
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')



# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     Dense(128, activation='relu'),
#     Dense(1)
# ])
# model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
# model.fit(X_train, y_train_scaled, epochs=50, validation_split=0.1)


learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    print(f"\nEvaluating model with learning rate: {lr}")
    model = Sequential([
        Dense(138, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(138, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    # Entraînement du modèle
    history=model.fit(X_train, y_train_scaled, epochs=50, validation_split=0.1, verbose=0)
    import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to calculate and print evaluation metrics
def evaluate_model(model, X_train, y_train, X_test, y_test, scaler_y, verbose=True):
    # Making predictions
    train_preds_scaled = model.predict(X_train)
    test_preds_scaled = model.predict(X_test)

    # Inversing scaling to obtain actual values
    train_preds = scaler_y.inverse_transform(train_preds_scaled)
    test_preds = scaler_y.inverse_transform(test_preds_scaled)
    y_train_actual = scaler_y.inverse_transform(y_train)
    y_test_actual = scaler_y.inverse_transform(y_test)

    # Calculating metrics
    train_mae = mean_absolute_error(y_train_actual, train_preds)
    test_mae = mean_absolute_error(y_test_actual, test_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_preds))

    if verbose:
        # print(f"Training MAE: {train_mae:.3f}")
        # print(f"Test MAE: {test_mae:.3f}")
        print(f"Training RMSE: {train_rmse:.3f}")
        print(f"Test RMSE: {test_rmse:.3f}")

    return train_mae, test_mae, train_rmse, test_rmse

# Use the model fitting and evaluation function in your learning rate loop
learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    print(f"\nEvaluating model with learning rate: {lr}")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    # Training the model
    history = model.fit(X_train, y_train_scaled, epochs=50, validation_split=0.1, verbose=0)

    # Evaluating the model
    evaluate_model(model, X_train, y_train_scaled, X_test, y_test_scaled, scaler_y)

    # Plotting training history
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss History at LR={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Évaluation du modèle sur les données de test
    test_loss = model.evaluate(X_test, y_test_scaled, verbose=0)
    print("Test loss:", test_loss)

    # Évaluation du modèle sur les données d'entraînement
    train_loss = model.evaluate(X_train, y_train_scaled, verbose=0)
    print("Training loss:", train_loss)

    # Calcul de MAE sur les données d'entraînement
    train_predictions_scaled = model.predict(X_train)
    train_predictions = scaler_y.inverse_transform(train_predictions_scaled)
    train_mae = mean_absolute_error(scaler_y.inverse_transform(y_train_scaled), train_predictions)
    print("Training MAE:", train_mae)

    # Prédiction sur les données de test
    test_predictions = model.predict(X_test)

    # Calcul de la RMSE pour les données de test
    test_rmse = np.sqrt(mean_squared_error(y_test_scaled, test_predictions))
    print("Test RMSE:", test_rmse)

 
    train_predictions = model.predict(X_train).flatten()
    train_rmse = np.sqrt(mean_squared_error(y_train_scaled, train_predictions))
    print("Training RMSE:", train_rmse)

if lr == 0.01:
        model.save('my_model_0_01.keras')
        print("Model 0.01 has been saved.")

# Affichage de la distribution de la variable 'consommation'
plt.figure(figsize=(10, 5))
plt.hist(data['consommation'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution de la consommationt')
plt.xlabel('Consommation')
plt.ylabel('Nombre d\'observations')
plt.show()



plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
#plt.plot(history.history['mean_squared_error'], label='Training MSE')
#plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.title('Model Training Metrics')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()


test_predictions = model.predict(X_test).flatten()

# Comparaison visuelle
plt.figure(figsize=(10, 5))
plt.plot(y_train_scaled, label='Valeurs Réelles')
plt.plot(train_predictions, label='Prédictions', alpha=0.7)
plt.title('Comparaison des valeurs réelles et prédites')
plt.xlabel('Échantillons')
plt.ylabel('Consommation')
plt.legend()
plt.show()

model.save('my_model.keras')






