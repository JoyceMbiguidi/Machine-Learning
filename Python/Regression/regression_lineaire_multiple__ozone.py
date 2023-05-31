#============
# REGRESSION LINEAIRE SIMPLE
# Objectif : expliquer et prédire les valeurs d'une feature
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
import pygwalker as pyg
gwalker = pyg.walk(ozone_df)

#============ matrice de correlation
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = ozone_df.drop(['vent', 'pluie'], axis=1).corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

#============ variable explicative
x = ozone_df['T12'].to_numpy()
x.shape

sns.histplot(data = x)

#============ variable à expliquer
y = ozone_df['maxO3'].to_numpy()
y.shape

sns.histplot(data = y)

"""
Problematique : On veut savoir s'il y a un lien entre la concentration d'ozone et la température à un moment donné de la journée
"""

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire

#============ MODELE 1 - entrainement du modèle
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))

#============ évaluation du modèle sur le jeu d'entrainement
x_train = x_train.reshape(-1,1)
y_predict_train = model1.predict(x_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

y_predict_train = model1.predict(x_train.reshape(-1,1))
mse_train = mean_squared_error(y_train, y_predict_train)
print("mse =", mse_train)

rmse_train = mean_squared_error(y_train, y_predict_train, squared = False)
print("rmse =", rmse_train)

r2_train = r2_score(y_train, y_predict_train)
print("R² =", r2_train)

#============ MODELE 1 - évaluation du modèle sur le jeu de test
x_test = x_test.reshape(-1,1)
y_predict_test = model1.predict(x_test)

#============ RMSE
RMSE_test = mean_squared_error(y_test, y_predict_test, squared = False)
RMSE_test

#============ coefficient d'ajustement (R²)
R_squared_test = r2_score(y_test, y_predict_test)
R_squared_test

"""Peut-on améliorer le R² ? Si oui, de quelle(s) façon(s) ?"""

#============ MODELE 2 - entrainement du modèle
# on recherche des individus extremes
import plotly.express as px
fig = px.box(ozone_df, y="T12")
fig.show()

#============ affichage des valeurs extrêmes
ozone_df[ozone_df["T12"]>= 32.7]

#============ supprimons les lignes 21 et 79 dans le jeu de données initial
ozone_df = ozone_df.drop(index=[21, 79])

#============ on vérifie la dimension du nouveau tableau
ozone_df.shape

#============ variable explicative
x = ozone_df['T12'].to_numpy()
x.shape

#============ variable à expliquer
y = ozone_df['maxO3'].to_numpy()
y.shape

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire

#============ entrainement du modèle 2
model2 = LinearRegression()
model2.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))

y_predict_train = model2.predict(x_train.reshape(-1,1))

#============ évaluation du modèle sur le jeu d'entrainement et comparaison
rmse_train2 = mean_squared_error(y_train, y_predict_train, squared = False)
print("rmse_2 =", rmse_train, " vs ", "rmse_2 = ", rmse_train2)

r2_train2 = r2_score(y_train, y_predict_train)
print("R²_1 =", r2_train, " vs ", "R²_2 = ", r2_train2)

#============ MODELE 2 - évaluation du modèle sur le jeu de test
x_test = x_test.reshape(-1,1)
y_predict_test = model2.predict(x_test)

#============ RMSE
RMSE_test = mean_squared_error(y_test, y_predict_test, squared = False)
RMSE_test

#============ coefficient d'ajustement (R²)
R_squared_test = r2_score(y_test, y_predict_test)
R_squared_test

"""Le modèle 2 n'est pas meilleur que le modèle 1.
Il semblerait que la suppression des valeurs extrêmes aient engendré des efforts de bords
"""










