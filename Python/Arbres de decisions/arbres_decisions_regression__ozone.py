#============
# ARBRES DE DECISIONS POUR LA REGRESSION
# Objectif : expliquer et prédire les valeurs de plusieurs feature
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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/ozone.csv"
raw_df = pd.read_csv(path, sep = ";", decimal = ",")
raw_df.head()

#============ sélection des variables
categ_feat = raw_df.select_dtypes(include='object')
categ_feat = pd.get_dummies(categ_feat, drop_first=True)
categ_feat.head()

numeric_feat = raw_df.select_dtypes(include='number').drop('obs', axis=1) 

#============ concatenation des variables
frames = [numeric_feat, categ_feat]
ozone_df = pd.concat(frames, axis=1)
ozone_df.head()

#============ vérification des types
ozone_df.dtypes

#============ afficher la dimension
print(ozone_df.shape)

#============ controle des valeurs manquantes
print(ozone_df.isna().sum())

#============ matrice de correlation
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = ozone_df.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

"""
Problematique : Quels sont les éléments météorologiques qui ont une influence sur la concentration maximale d'ozone dans la journée ?
"""

#============ variables explicatives
x = ozone_df.drop(['maxO3'], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = ozone_df['maxO3'].to_numpy()
y.shape

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire

#============ MODELE 1 - entrainement du modèle
model1 = DecisionTreeRegressor(random_state = 42, max_depth = 5)
model1.fit(x_train, y_train)

#============ visuel de l'arbre
from sklearn.tree import plot_tree
plt.figure(figsize=(20,20))
plot_tree(model1, feature_names=list(ozone_df.drop(['maxO3'], axis=1).columns), filled=True)
pass

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

#============ feature importance
importances_sk = model1.feature_importances_

#============ dataframe de l'importance des variables
# on recupere le nom des variables 
features = ozone_df.drop(['maxO3'], axis=1).columns.tolist()

feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)


feat_importance_df = pd.DataFrame(feature_importance_sk.items(), columns=['Variables', 'Importance']).sort_values('Importance', ascending=False)
feat_importance_df

""" 
on remarque un overfitting dans le jeu d'entrainement.
Le modele ne semble pas bien se generaliser avec ce jeu de donnees.
Un autre algorithme peut mieux faire. Il faudra le trouver...
"""