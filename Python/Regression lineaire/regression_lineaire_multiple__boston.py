#============
# REGRESSION LINEAIRE MULTIPLE
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

#============ elements graphiques
# pip install pygwalker
import pygwalker as pyg
gwalker = pyg.walk(df)

#============ matrice de correlation
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = df.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

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
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)

#============ MODELE 1 sur données non standardisées
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(x_train, y_train)

#============ prédictions sur les jeux d'entrainement et de test
y_predict_train = model1.predict(x_train)
y_predict_test = model1.predict(x_test)

#============ évaluation du modèle sur les jeux d'entrainement et de test
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

rmse_train = mean_squared_error(y_train, y_predict_train, squared = False)
r2_train = r2_score(y_train, y_predict_train)

rmse_test = mean_squared_error(y_test, y_predict_test, squared = False)
r2_test = r2_score(y_test, y_predict_test)

#============ valeurs reelles vs valeurs predites
scores = {}
scores['realite'] = (y_train,
                  y_test)
scores['prediction'] = (y_predict_train.reshape(-1,1).round(2),
                y_predict_test.reshape(-1,1).round(2))
scores_df = pd.DataFrame(scores).transpose()
scores_df.columns = ['Train', 'Test']
print(scores_df)

#============ table des métriques
metrics = {}
metrics['r2'] = (r2_train.round(3),
                  r2_test.round(3))
metrics['rmse'] = (rmse_train,
                  rmse_test)
metrics_df = pd.DataFrame(metrics).transpose()
metrics_df.columns = ['Train', 'Test']
print(metrics_df)

"""
Peut-on améliorer le R² ? Si oui, de quelle(s) façon(s) ?
"""

#============ MODELE 2 avec stepwise
#============ statsmodels avec toutes les features du jeu d'entrainement
import statsmodels.formula.api as smf
from stepwise_regression import step_reg

x_train_df = pd.DataFrame(x_train, columns=['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'])
y_train_df = pd.DataFrame(y_train, columns=['medv'])

model_smf = smf.ols(formula='y_train ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + b + lstat', data = x_train_df).fit()

#============ resultats de statsmodels
print(model_smf.summary())

#============ model avec stepwise
backselect = step_reg.backward_regression(x_train_df, y_train_df, 0.05, verbose=True) # 0.05 est la valeur seuil p-value
backselect

#============ nouveau modèle après stepwise
x_train_clean = x_train_df.drop(['age', 'indus', 'zn'], axis=1) # on retire les variables dont les p-values sont trop élevées

model_smf = smf.ols(formula='y_train_df ~ crim + chas + nox + rm + dis + rad + tax + ptratio + b + lstat', data = x_train_clean).fit()
print(model_smf.summary())

#============ statsmodels avec toutes les features du jeu de test
# on passe de numpy array à un DataFrame
x_test_df = pd.DataFrame(x_test, columns=['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'])
y_test_df = pd.DataFrame(y_test, columns=['medv'])

model_smf = smf.ols(formula='y_test_df ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + b + lstat', data = x_test_df).fit()

#============ resultats de statsmodels
print(model_smf.summary())

#============ model avec stepwise
backselect = step_reg.backward_regression(x_test_df, y_test_df, 0.05, verbose=True) # 0.05 est la valeur seuil p-value
backselect

#============ nouveau modèle après stepwise
x_test_clean = x_test_df.drop(['indus', 'crim', 'rm', 'chas', 'age', 'tax'], axis=1) # on retire les variables dont les p-values sont trop élevées

model_smf = smf.ols(formula='y_test_df ~ zn + nox + dis + rad + ptratio + b + lstat', data = x_test_clean).fit()
print(model_smf.summary())

