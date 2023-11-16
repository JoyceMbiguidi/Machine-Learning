#============
# EXTREME GRADIENT BOOSTING  POUR LA REGRESSION
# Objectif : expliquer et prédire les valeurs de plusieurs features
#============

#============ description des données
"""
Jeu de données sur la pollution de l'air. Nous disposons de 112 relevés durant l'été 2001 à Rennes.

maxO3 = maximum journalier de la concentration en ozone (micro grammes / m3)
T9 = température à 9h
T12 = température à 12h
T15 = température à 15h
Ne9 = nébulosité à 9h
Ne12 = nébulosité à 12h
Ne15 = nébulosité à 15h
Vx9 = projection du vent sur l'axe Est-Ouest à 9h
Vx12 = projection du vent sur l'axe Est-Ouest à 12h
Vx15 = projection du vent sur l'axe Est-Ouest à 15h
maxO3v
vent
pluie
"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/ozone.csv"
raw_df = pd.read_csv(path, sep = ";", decimal = ",")

#============ copie du dataset brut
ozone_df = raw_df
ozone_df.head()

#============ vérification des types
ozone_df.dtypes

#============ afficher la dimension
print(ozone_df.shape)

#============ elements graphiques
# pip install pygwalker
#import pygwalker as pyg
#gwalker = pyg.walk(ozone_df)

#============ matrice de correlation
corr_matrix = ozone_df.drop(['vent', 'pluie'], axis=1).corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

"""
Problematique : On veut savoir s'il y a un lien entre la concentration d'ozone et la température à un moment donné de la journée
"""

#============ variable explicative
x = ozone_df.drop(['maxO3', 'vent', 'pluie'], axis=1).to_numpy()
x.shape


#============ variable à expliquer
y = ozone_df['maxO3'].to_numpy()
y.shape

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire

#============ MODELE 1 - entrainement du modèle
# instantiation
xgb_r = ensemble.GradientBoostingRegressor(n_estimators = 300, criterion="squared_error")

# ajustement du modèle
xgb_r.fit(x_train, y_train)

# prediction sur ensemble de test
y_pred = xgb_r.predict(x_test)

# MSE
mse = mean_squared_error(y_test, y_pred)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

# R²
r_squared = xgb_r.score(x_test, y_test)
print("R squared on test set: {:.4f}".format(r_squared))

# deviance de l'ensemble d'entrainement
"""
La déviance est une mesure de l'ajustement du modèle aux données observées. 
Elle est calculée en comparant les prédictions du modèle avec les valeurs réelles de la variable cible
"""

test_score = np.zeros((300,), dtype=np.float64)
for i, y_pred in enumerate(xgb_r.staged_predict(x_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(300) + 1,
    xgb_r.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(300) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()

# importance des variables
feature_importance = xgb_r.feature_importances_

sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(ozone_df.drop(['maxO3', 'vent', 'pluie'], axis=1).columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    xgb_r, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(ozone_df.drop(['maxO3', 'vent', 'pluie'], axis=1).columns)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()


