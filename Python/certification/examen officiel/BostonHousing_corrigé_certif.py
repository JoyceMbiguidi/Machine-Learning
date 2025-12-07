# Exercice : Analyse du Boston Housing Dataset
"""
1. Présentation des données brutes :
    Boston Housing Dataset contient des informations sur les prix des maisons à Boston, 
    avec plusieurs caractéristiques qui influencent ces prix. Nous allons charger et afficher les premières lignes du dataset.
"""
# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Chargement des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/Boston_dataset.csv"
raw_df = pd.read_csv(path, sep = ";")

# Affichage des premières lignes des données
print(raw_df.head())


"""
2. Justification des choix lors du prétraitement des données
    Avant de procéder à l'analyse, nous devons vérifier la présence de valeurs manquantes et effectuer un éventuel encodage des variables catégorielles. 
    Dans ce cas, il n'y a pas de variables catégorielles, mais nous allons vérifier les valeurs manquantes.
"""
# Vérification des valeurs manquantes
print(raw_df.isnull().sum())

"""
Aucune valeur manquante.
Dans ce cas, nous n'avons pas besoin de prétraitement supplémentaire, car les données sont déjà propres.

3. Étapes et choix méthodologiques
    Nous allons diviser les données en ensembles d'entraînement et de test, puis entraîner un modèle de régression linéaire. Voici les étapes :
        - Séparation des données : Diviser le dataset en ensembles d'entraînement (80%) et de test (20%).
        - Entraînement du modèle : Utiliser la régression linéaire pour prédire les prix des maisons.
        - Évaluation du modèle : Calculer les métriques de performance.
"""
# Séparation des données
X = raw_df.drop(['medv'], axis=1).to_numpy()
y = raw_df['medv'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

"""
4. Analyse des métriques et coefficients associés
    Nous allons maintenant évaluer les performances du modèle à l'aide de l'erreur quadratique moyenne (RMSE) et du coefficient de détermination (R²).
"""
# Calcul des métriques
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Coefficients du modèle
coefficients = pd.DataFrame(model.coef_, raw_df.drop(['medv'], axis=1).columns, columns=['Coefficient'])
print(coefficients)

"""
    - RMSE : Mesure l'erreur moyenne des prédictions. Plus il est bas, mieux c'est.
    - R² : Indique la proportion de la variance des prix expliquée par le modèle. Un R² proche de 1 signifie un bon ajustement.
"""

"""
5. Recommandations métiers
    - Investir dans des zones avec des caractéristiques positives : Les coefficients positifs des variables telles que RM (nombre de pièces) et LSTAT (pourcentage de population à faible statut) peuvent indiquer qu'une augmentation de ces caractéristiques pourrait augmenter les prix.
    - Prendre en compte les variables négatives : Les variables avec des coefficients négatifs, comme NOX (concentration d'oxyde d'azote), doivent être surveillées, car elles peuvent avoir un impact défavorable sur les prix des maisons.
    - Évaluer l'impact des facteurs externes : Considérer d'autres facteurs comme l'accessibilité aux transports et la qualité des écoles, qui peuvent influencer les prix de manière significative.
"""