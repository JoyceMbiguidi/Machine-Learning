#============
# RANDOM FOREST POUR LA REGRESSION
# Objectif : expliquer et prédire les valeurs de plusieurs features
#============

#============ description des données
"""
CRIM : taux de criminalité par habitant et par commune.
ZN : proportion des terrains résidentiels zonés pour les terrains de plus de 25 000 pieds carrés, soit 2322.576 m².
INDUS : proportion d'acres d'entreprises non commerciales par ville.
CHAS : variable dummy Charles River : 1 = si le terrain délimite la rivière, 0 sinon.
NOX : concentration d'oxydes nitriques.
RM : nombre moyen de chambres par logement.
AGE : proportion de logements occupés par leur propriétaire construits avant 1940.
DIS : distances pondérées à cinq centres d'emploi de Boston.
RAD : indice d'accessibilité aux autoroutes (rayon).
TAX : taux de la taxe foncière sur la valeur totale par 10 000 dollars.
PTRATIO : ratio d'élèves-enseignant par ville.
B : indice de proportion de population noires par villes.
LSTAT : pourcentage de statut inférieur de la population.
MEDV : PRIX ou valeur médiane des maisons occupées par leur propriétaire en milliers de dollars.

"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/Boston_dataset.csv"
raw_df = pd.read_csv(path, sep = ";")

#============ copie du dataset brut
df = raw_df
df.head()

#============ vérification des types
df.dtypes

#============ afficher la dimension
print(df.shape)

#============ check des valeurs manquantes
df.isna().sum()

"""
Problematique : on veut expliquer et prédire les facteurs qui ont une influence sur le prix des maisons
"""

#============ variables explicatives
x = df.drop(['medv'], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = df['medv'].to_numpy()
y.shape

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)

#============ MODELE 1
model1 = RandomForestRegressor()
model1.fit(x_train, y_train)

#============ prediction sur le jeu de test
y_pred = model1.predict(x_test)

#============ R2 coefficient de determination
r2 = r2_score(y_test, y_pred).round(3)
print(r2)

#============ RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
print("\nRMSE: ", rmse)

#============ importance des variables
# dictionnaire des features + importance des variables
feat_dict= {}
for col, val in sorted(zip(df.drop('medv', axis = 1).columns, model1.feature_importances_), key=lambda x:x[1], reverse=True):
  feat_dict[col]=val

# on convertit le dictionnaire en dataframe
feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})
feat_df


#============
# Hyperparameter tunning
#============

hyperparameters = {'max_depth':range(3,21),
                   'n_estimators': [10, 20, 30, 40, 50, 60, 80, 100, 120, 150],
                   'n_jobs': [-1]}

model2 = RandomForestRegressor(random_state = 42, n_estimators = 5)

gridsearchcv = GridSearchCV(model2, hyperparameters, cv = 5)

gridsearchcv.fit(x_train,y_train)

gridsearchcv.best_params_

#============ nouveau modele
model_rf = RandomForestRegressor(random_state = 42, n_estimators = 100, max_depth = 12)
model_rf.fit(x_train, y_train)

y_predict_train = model_rf.predict(x_train)
y_predict_test = model_rf.predict(x_test)

r_score_train = r2_score(y_train, y_predict_train)
r_score_test = r2_score(y_test, y_predict_test)

print(r_score_train)
print(r_score_test)

#============ importance des variables
# dictionnaire des features + importance des variables
feat_dict= {}
for col, val in sorted(zip(df.drop('medv', axis = 1).columns, model1.feature_importances_), key=lambda x:x[1], reverse=True):
  feat_dict[col]=val

# on convertit le dictionnaire en dataframe
feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})
feat_df



