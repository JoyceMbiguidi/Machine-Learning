# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Chargement des données IRIS
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/iris.csv"
iris_df = pd.read_csv(path, sep = "|", encoding = 'tuf-8')

# Imputation des valeurs manquantes
iris_df.fillna(iris_df.mean(), inplace=True)

# Séparation des features et de la cible
X = iris_df.drop("species", axis=1)
y = iris_df["species"]

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop('species', axis=1), iris_df['species'], test_size=0.5, random_state=42)

# Création du modèle de classification
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred = model.predict(X_test)

# Calcul de l'accuracy
accuracy = accuracy_score(y_test, y_pred)

# Affichage des résultats
print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))