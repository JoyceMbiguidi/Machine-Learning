## Code python corrigé

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Chargement des données IRIS
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/iris.csv"
iris_df = pd.read_csv(path, sep = ";", encoding = 'utf-8')

# Séparation des features et de la cible
X = iris_df.drop("Species", axis=1)
y = iris_df["Species"]

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
print("Classification Report: \n", classification_report(y_test, y_pred))


"""
Utilisation correcte de train_test_split :
    Les données X et y sont directement utilisées sans conversion en DataFrame, ce qui simplifie le code.

Séparation des données :
    La séparation des données est maintenant effectuée correctement, en utilisant directement les matrices X et y.

Ajout de métriques d'évaluation :
    Un rapport de classification a été ajouté avec classification_report, qui fournit des informations sur la précision, le rappel et le score F1.

Affichage des résultats :
    L'affichage des résultats est plus complet et clair grâce à l'ajout du rapport de classification.
"""