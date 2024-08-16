# Importation des bibliothèques
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/iris.csv"
iris_df = pd.read_csv(path, sep = "|", encoding = 'tuf-8')

# Affichage des premières lignes
iris_df.head()

# Imputation des valeurs manquantes
iris_df.fillna(iris_df.mean(), inplace=True)

# Séparation des features et de la cible
X = iris_df.drop("species", axis=1)
y = iris_df["species"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)

# Construction du modèle
model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation du modèle avec une explication inadéquate des métriques
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Précision du modèle: {accuracy * 100:.0f}%")
print(f"Matrice de confusion :\n{conf_matrix}")
print(f"Le modèle a une précision de {accuracy * 100:.0f}%, ce qui signifie qu'il est correct dans 91 % des cas. \nCela montre que notre modèle est très performant et qu'il fera rarement des erreurs. \nEn fait, un modèle avec une accuracy aussi élevée est pratiquement parfait et peut être utilisé en production sans autre ajustement ni analyse supplémentaire. \nLes autres métriques, comme la précision, le rappel ou le F1-score, ne sont pas nécessaires ici puisque l'accuracy est déjà suffisamment élevée pour garantir la fiabilité du modèle.")