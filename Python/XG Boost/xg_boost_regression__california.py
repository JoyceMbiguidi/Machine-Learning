#============
# EXTREME GRADIENT BOOSTING POUR LA REGRESSION
# Objectif : expliquer et prédire les valeurs de plusieurs features
#============

#============ description des données
"""
La variable cible est la valeur médiane des maisons des districts de Californie,
exprimé en centaines de milliers de dollars (100 000 $).

Cet ensemble de données provient du recensement américain de 1990, en utilisant une ligne par recensement
groupe de bloc. Un groupe de blocs est la plus petite unité géographique pour laquelle
Le Bureau Census des États-Unis publie des exemples de données (un groupe d'îlots a généralement une population
de 600 à 3 000 personnes).

Un ménage est un groupe de personnes résidant au sein d'un logement. La moyenne
du nombre de pièces et de chambres à coucher sont fournies par ménage, ces
les colonnes peuvent prendre des valeurs étonnamment élevées pour les groupes d'îlots avec peu de ménages
et de nombreuses maisons vides, telles que des centres de villégiature.

MedInc : revenu médian dans le groupe de blocs
HouseAge : âge médian de la maison dans le groupe d'îlots
AveRooms : nombre moyen de pièces par ménage     
AveBedrms : nombre moyen de chambres par ménage    
Population : population du groupe de blocs   
AveOccup :  nombre moyen de membres du ménage
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
from sklearn.datasets import fetch_california_housing
raw_df = fetch_california_housing(return_X_y = False)
print(raw_df.DESCR)

#============ variable explicative
df_ = pd.DataFrame(raw_df.data, columns = raw_df.feature_names) #.data signifie variables explicatives
df_.head()

#============ variable à expliquer
df_['Price'] = raw_df.target # targetnumpy array of shape (20640,).
df_.head()

#============ recherche des valeurs manquantes
print(df_.isnull().sum())

#============ copie du dataset brut
df = df_
df.head()

#============ matrice de correlation
corr_matrix = df.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

"""
Problematique : on veut expliquer et prédire les facteurs qui ont une influence sur le prix des logements
"""

#============ variable à expliquer et features
x = df.drop(['Price', 'Latitude', 'Longitude'], axis=1).to_numpy() # feature // variable(s) indépendante(s) // variable(s) explicative(s)
y = df['Price'].to_numpy() # target, i.e. la variable cible // variable dépendante // variable à expliquer

#============ standardisation des variables explicatives
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

X = x
scaledX = scale.fit_transform(X)
scaledX # on obtient un array qu'il faut convertir en DataFrame
scaledX_df = pd.DataFrame(scaledX, columns = ['MedInc_', 'HouseAge_', 'AveRooms_', 'AveBedrms_', 'Population_', 'AveOccup_'])
scaledX_df.head()

#============ on travaille désormais avec les données standardisées
df = pd.concat([df_['Price'], scaledX_df], axis = 1) # on merge les deux DF
df.head()

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(scaledX_df, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire
x_train.shape
y_train.shape
x_test.shape
y_test.shape

#============ MODELE 1 - entrainement du modèle
# instantiation
xgb_r = ensemble.GradientBoostingRegressor(n_estimators = 100, criterion="squared_error")

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

test_score = np.zeros((100,), dtype=np.float64)
for i, y_pred in enumerate(xgb_r.staged_predict(x_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(100) + 1,
    xgb_r.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(100) + 1, test_score, "r-", label="Test Set Deviance"
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
plt.yticks(pos, np.array(scaledX_df.columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    xgb_r, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(scaledX_df.columns)[sorted_idx],
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


