#============
# REGRESSION LINEAIRE MULTIPLE
# Objectif : expliquer et prédire les valeurs de plusieurs features
#============

#============ description des données
"""
La bière est l'une des boissons les plus démocratiques et les plus consommées au monde.
L'objectif de ce travail sera de démontrer les impacts des variables sur la consommation de bière.
Les données (échantillon) ont été recueillies à São Paulo au Brésil, dans une zone universitaire, 
où il y a des soirées avec des groupes d'étudiants de 18 à 28 ans (moyenne).

Les variables sont explicites.
"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import statsmodels.api as sm
import statsmodels.formula.api as smf
from stepwise_regression import step_reg

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/consumo_cerveja.csv"
raw_df = pd.read_csv(path, sep = ",", decimal = ',')

#============ copie du dataset brut
df = raw_df
df.head()

#============ vérification des types
df.info()

"""
la variable "consomu de cerveja" est curieusement de type qualitative. Transformons-la.
"""
df['Consumodecerveja(litros)'] = pd.to_numeric(df['Consumodecerveja(litros)'])

#============ afficher la dimension
print(df.shape)

#============ vérification des valeurs manquantes
print(df.isnull().sum())

#============ suppression des valeurs manquantes
df = df.dropna()
df.shape

#============ matrice de correlation
df = df.drop('Data', axis = 1)
corr_matrix = df.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (13,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()
   
"""
Problematique : à Sao Paulo, quels facteurs ont un impact sur la consommation de bières ?
"""

#============ variables explicatives
x = df.drop(['Consumodecerveja(litros)'], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = df['Consumodecerveja(litros)'].to_numpy()
y.shape

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)

#============ statsmodels avec toutes les features du jeu d'entrainement
# on passe de numpy array à un DataFrame
x_train_df = pd.DataFrame(x_train, columns=['TemperaturaMedia(C)', 'TemperaturaMinima(C)', 'TemperaturaMaxima(C)', 'Precipitacao(mm)', 'FinaldeSemana'])
y_train_df = pd.DataFrame(y_train, columns=['Consumodecerveja(litros)'])

model_smf = smf.ols(formula='y_train_df ~ x_train_df', data = x_train_df).fit()

#============ resultats de statsmodels
print(model_smf.summary())

#============ model avec stepwise
backselect = step_reg.backward_regression(x_train_df, y_train_df, 0.05, verbose=True) # 0.05 est la valeur seuil p-value
backselect

#============ nouveau modèle après stepwise
x_train_clean = x_train_df.drop(['TemperaturaMedia(C)', 'TemperaturaMinima(C)'], axis=1) # on retire les variables dont les p-values sont trop élevées

model_smf = smf.ols(formula='y_train_df ~ x_train_clean', data = x_train_clean).fit()
print(model_smf.summary())

#============ statsmodels avec toutes les features du jeu de test
# on passe de numpy array à un DataFrame
x_test_df = pd.DataFrame(x_test, columns=['TemperaturaMedia(C)', 'TemperaturaMinima(C)', 'TemperaturaMaxima(C)', 'Precipitacao(mm)', 'FinaldeSemana'])
y_test_df = pd.DataFrame(y_test, columns=['Consumodecerveja(litros)'])

model_smf = smf.ols(formula='y_test_df ~ x_test_df', data = x_test_df).fit()

#============ resultats de statsmodels
print(model_smf.summary())

#============ model avec stepwise
backselect = step_reg.backward_regression(x_test_df, y_test_df, 0.05, verbose=True) # 0.05 est la valeur seuil p-value
backselect

#============ nouveau modèle après stepwise
x_test_clean = x_test_df.drop(['TemperaturaMaxima(C)', 'Precipitacao(mm)', 'FinaldeSemana'], axis=1) # on retire les variables dont les p-values sont trop élevées

model_smf = smf.ols(formula='y_test_df ~ x_test_clean', data = x_test_clean).fit()
print(model_smf.summary())

