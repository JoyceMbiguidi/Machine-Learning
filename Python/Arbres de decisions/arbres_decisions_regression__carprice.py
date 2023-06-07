#============
# ARBRES DE DECISIONS
# Objectif : expliquer et prédire les valeurs de plusieurs feature
#============

#============ description des données
"""
Prédiction du prix de voitures américaines en fonction de leurs caractéristiques

make
model
priceUSD
year
condition
mileage(kilometers)
fuel_type
volume(cm3)
color
transmission
drive_unit
segment
"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import numpy as np
from sklearn.base import r2_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import itertools
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/cars.csv"
raw_df = pd.read_csv(path, sep = ",")
raw_df.head()

#============ controle des valeurs manquantes
raw_df.isna().sum()
raw_df = raw_df.dropna()
raw_df.head()

#============ controle des variables categorielles
raw_df.nunique()

#============ pre-sélection des variables
raw_df = raw_df.drop(['model', 'year'], axis=1)

#============ sélection des variables
categ_feat = raw_df.select_dtypes(include='object')
categ_feat = pd.get_dummies(categ_feat, drop_first=True)
categ_feat.head()

numeric_feat = raw_df.select_dtypes(include='number')
numeric_feat.head()

#============ concatenation des variables
frames = [numeric_feat, categ_feat]
cars_df = pd.concat(frames, axis=1)
cars_df.head()

#============ matrice de correlation
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = cars_df.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

"""
Problematique : Quelles caractéristiques mécaniques et esthétiques ont une influence sur le prix des véhicules américains ?
"""

#============ variables explicatives
x = cars_df.drop(['priceUSD'], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = cars_df['priceUSD'].to_numpy()
y.shape

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire

#============ MODELE 1 - entrainement du modèle
model1 = DecisionTreeRegressor(random_state = 42, max_depth = 5)
model1.fit(x_train, y_train)

#============ prediction des donnees de test
y_pred = model1.predict(x_test)

#============ evaluation des metriques
score_train = model1.score(x_train, y_train).round(3)
score_test = model1.score(x_test, y_test).round(3)

print("R² train :", score_train)
print("R² test :", score_test)

#============ mean absolute error (mse) 
mae_train = mean_absolute_error(y_train, model1.predict(x_train)).round(2)
mae_test = mean_absolute_error(y_test, y_pred).round(2)

print(f"Mean absolute error on training set: {mae_train}")
print(f"Mean absolute error on test set: {mae_test}")

plt.figure(figsize=(20,9))
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="original")
plt.plot(x_ax, y_pred, linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data : R² = {}".format(score_test))
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()

""" 
pas d'overfitting, les R2 sont corrects et les erreurs sont quasi similaires.
peut-on encore améliorer ce modèle ?
"""

#============
# Hyperparameter tunning
#============
hyperparameter = {'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
model2 = DecisionTreeRegressor(random_state = 42)

gridsearchcv = GridSearchCV(model2, hyperparameter, cv = 5)
gridsearchcv.fit(x_train, y_train)

#============ affichage du meilleur hyper parametre
gridsearchcv.best_params_

#============ evaluation du nouveau modele
model_tree = DecisionTreeRegressor(random_state = 42, max_depth = 14)
model_tree.fit(x_train,y_train)

y_predict_train = model_tree.predict(x_train)
y_predict_test = model_tree.predict(x_test)

r_score_train = r2_score(y_train, y_predict_train).round(3)
r_score_test = r2_score(y_test, y_predict_test).round(3)

print(r_score_train)
print(r_score_test)

""" l'utilisation des hyper parametre a dégradé les performances du model. 
Soit on revient au modèle performant, soit on tatonne avec les hyper parametres 
dans l'espoir d'avoir un meilleur modele
"""


#============ feature importance du meilleur modele (initial)
importances_sk = model1.feature_importances_

#============ dataframe de l'importance des variables
# on recupere le nom des variables 
features = cars_df.drop(['priceUSD'], axis=1).columns.tolist()

feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)


feat_importance_df = pd.DataFrame(feature_importance_sk.items(), columns=['Variables', 'Importance']).sort_values('Importance', ascending=False)
feat_importance_df