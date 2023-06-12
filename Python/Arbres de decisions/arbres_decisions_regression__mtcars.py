#============
# ARBRES DE DECISIONS POUR LA REGRESSION
# Objectif : expliquer et prédire les valeurs de plusieurs feature
#============

#============ description des données
"""
MTCARS :
	The data was extracted from the 1974 Motor Trend US magazine, 
	and comprises fuel consumption and 10 aspects of automobile design 
	and performance for 32 automobiles (1973–74 models).

mpg 	Miles/(US) gallon
cyl 	Number of cylinders
disp 	Displacement (cu.in.)
hp 	Gross horsepower
drat 	Rear axle ratio
wt 	Weight (lb/1000)
qsec 	1/4 mile time
vs 	V/S
am 	Transmission (0 = automatic, 1 = manual)
gear 	Number of forward gears
carb 	Number of carburetors
"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import itertools
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/mtcars.csv"
raw_df = pd.read_csv(path, sep = ",")

#============ copie du dataset brut
mtcars_df = raw_df
mtcars_df.head()

#============ vérification des types
mtcars_df.dtypes

#============ afficher la dimension
print(mtcars_df.shape)

#============ matrice de correlation
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = mtcars_df.drop(['model'], axis=1).corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

"""
Problematique : quelles caractéristiques du véhicule ont un impact sur la consommation de carburant ?
"""

#============ variables explicatives
x = mtcars_df.drop(["mpg", "model"], axis= 1).to_numpy()
x.shape

#============ variable à expliquer
y = mtcars_df['mpg'].to_numpy()
y.shape

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)

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

#============ feature importance modele 1
plt.barh(mtcars_df.drop(["mpg", "model"], axis= 1).columns, model1.feature_importances_)

""" 
peut-on ameliorer ce modele ?
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
model_tree = DecisionTreeRegressor(random_state = 42, max_depth = gridsearchcv.best_params_.get('max_depth'))
model_tree.fit(x_train,y_train)

y_predict_train = model_tree.predict(x_train)
y_predict_test = model_tree.predict(x_test)

r_score_train = r2_score(y_train, y_predict_train).round(3)
r_score_test = r2_score(y_test, y_predict_test).round(3)

print(r_score_train)
print(r_score_test)

#============ feature importance modele 2
plt.barh(mtcars_df.drop(["mpg", "model"], axis= 1).columns, model_tree.feature_importances_)

#============ feature importance du meilleur modele (initial)
importances_sk = model1.feature_importances_

#============ dataframe de l'importance des variables
# on recupere le nom des variables 
features = mtcars_df.drop(["mpg", "model"], axis= 1).columns.tolist()

feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)


feat_importance_df = pd.DataFrame(feature_importance_sk.items(), columns=['Variables', 'Importance']).sort_values('Importance', ascending=False)
feat_importance_df