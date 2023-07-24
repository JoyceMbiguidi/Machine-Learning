#============
# EXTREME GRADIENT BOOSTING POUR LA REGRESSION
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
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
#import pygwalker as pyg
#gwalker = pyg.walk(df)

#============ matrice de correlation
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)

#============ MODELE 1
"""
n_estimators : the number of boosting stages that will be performed. Later, we will plot deviance against boosting iterations.

max_depth : limits the number of nodes in the tree. The best value depends on the interaction of the input variables.

min_samples_split : the minimum number of samples required to split an internal node.

learning_rate : how much the contribution of each tree will shrink.

loss : loss function to optimize. The least squares function is used in this case however, there are many other options
"""

# instantiation
xgb_r = ensemble.GradientBoostingRegressor(n_estimators = 200, criterion="squared_error")

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

test_score = np.zeros((200,), dtype=np.float64)
for i, y_pred in enumerate(xgb_r.staged_predict(x_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(200) + 1,
    xgb_r.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(200) + 1, test_score, "r-", label="Test Set Deviance"
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
plt.yticks(pos, np.array(df.drop(['medv'], axis=1).columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    xgb_r, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(df.drop(['medv'], axis=1).columns)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()

""" peut-on ameliorer ce modele ?"""

#============ MODELE 2
params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

# instantiation
xgb_r2 = ensemble.GradientBoostingRegressor(**params)

# ajustement du modèle
xgb_r2.fit(x_train, y_train)

# prediction sur ensemble de test
y_pred = xgb_r.predict(x_test)

# MSE
mse = mean_squared_error(y_test, y_pred)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

# R²
r_squared = xgb_r.score(x_test, y_test)
print("R squared on test set: {:.4f}".format(r_squared))

""" il n'est plus possible d'améliorer ce modèle."""

