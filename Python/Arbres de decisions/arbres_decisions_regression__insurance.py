#============
# REGRESSION LINEAIRE MULTIPLE
# Objectif : expliquer et prédire les valeurs de plusieurs features
#============

#============ description des données
"""
	Le jeu de données insurance.csv contient des informations concernant des assurés et leurs frais de santé 
	(colonne expenses). L'objectif est de construire un modèle prédictif (regression linéaire multiple) 
	pour prédire ces frais pour mieux adapter le coût de l'assurance.


age: age of primary beneficiary
sex: insurance contractor gender, female, male
bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
    objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
children: Number of children covered by health insurance / Number of dependents
smoker: Smoking
region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
charges: Individual medical costs billed by health insurance
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
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/insurance.csv"
raw_df = pd.read_csv(path, sep = ",")

#============ copie du dataset brut
Insurance_df = raw_df
Insurance_df.head()

#============ vérification des types
Insurance_df.info()

#============ afficher la dimension
print(Insurance_df.shape)

#============ vérification des valeurs manquantes
print(Insurance_df.isnull().sum())

#============ matrice de correlation
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = Insurance_df.drop(['sex', 'smoker', 'region'], axis=1).corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()
   
"""
Problematique : quels éléments ont un impact sur le coût de l'assurance maladie ?
"""

#============ regroupement des variables par type
numeric_feats = ["age", "bmi", "children"] # variables numériques
categ_feats = ["sex", "smoker", "region"] # variables catégorielles
target = "expenses"

#============ dummy variables
df_fe = pd.get_dummies(Insurance_df, columns=categ_feats, drop_first=True)
df_fe.head()

#============ variables explicatives
x = df_fe.drop("expenses", axis = 1).to_numpy()
x.shape

#============ variable à expliquer
y = df_fe["expenses"].to_numpy()
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
plt.barh(df_fe.drop(["expenses"], axis= 1).columns, model1.feature_importances_)

""" 
peut-on ameliorer ce modele ?
"""

#============
# Hyperparameter tunning
#============

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
plt.barh(df_fe.drop("expenses", axis = 1).columns, model_tree.feature_importances_)

#============ feature importance du meilleur modele (initial)
importances_sk = model1.feature_importances_

#============ dataframe de l'importance des variables
# on recupere le nom des variables 
features = df_fe.drop("expenses", axis = 1).columns.tolist()

feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)


feat_importance_df = pd.DataFrame(feature_importance_sk.items(), columns=['Variables', 'Importance']).sort_values('Importance', ascending=False)
feat_importance_df
