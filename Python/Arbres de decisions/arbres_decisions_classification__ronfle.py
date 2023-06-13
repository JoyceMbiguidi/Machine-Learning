#============
# ARBRES DE DECISIONS POUR LA CLASSIFICATION
# Objectif : prédire les valeurs d'une target binaire et expliquer l'impact des features sur la target
#============

#============ description des données
"""
On fait une étude sur le ronflement. Nous avons des caractéristiques physiologiques et d'habitudes de consommation de quelques patients.

caractéristiques physiologiques : taille, poids, age, sexe...
habitudes de consommation : alcool, fumeur...
Problématique : On veut savoir ce qui provoque le ronflement autrement dit, quelles sont les facteurs responsables du ronflement ?
"""

#============ vérifier la propreté du code
# pip install flake8P
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/ronfle.txt"
raw_df = pd.read_csv(path, sep = "\t")

#============ copie du dataset brut
ronfle_df = raw_df
ronfle_df.head()

#============ vérification des types
ronfle_df.dtypes

#============ afficher la dimension
print(ronfle_df.shape)

#============ recherche des valeurs manquantes
ronfle_df.isnull().sum()

"""
Problematique : quels sont les facteurs responsables du ronflement ?
"""

#============ recodage des variables
ronfle_df.replace(('N', 'O'), (0, 1), inplace=True) # recodage des variables
ronfle_df.replace(('F', 'H'), (0, 1), inplace=True) # recodage des variables
ronfle_df.head()

#============ variables explicatives
x = ronfle_df.drop(['ronfle'], axis = 1)
x.shapes

#============ variable à expliquer
y = ronfle_df['ronfle']

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)

#============ MODELE 1 - entrainement du modèle
model1 = DecisionTreeClassifier(criterion='gini')
model1.fit(x_train, y_train)

#============ visuel de l'arbre
plt.figure(figsize=(12,12))
plot_tree(model1, feature_names=list(ronfle_df.columns), filled=True, class_names=str(model1.classes_))
pass

#============ predictions sur les donnees de test
y_predict_test = model1.predict(x_test)
accuracy_score(y_test, y_predict_test)

#============ matrice de confusion
cm_test = confusion_matrix(y_test, y_predict_test)
print(cm_test)

#============ importance des variables
feature_scores = pd.Series(model1.feature_importances_, index=ronfle_df.drop(['ronfle'], axis = 1).columns).sort_values(ascending=False)
feature_scores












