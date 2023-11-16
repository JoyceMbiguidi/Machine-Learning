#============
# EXTREME GRADIENT BOOSTING POUR LA REGRESSION
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
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

#============ elements graphiques
# pip install pygwalker
#import pygwalker as pyg
#gwalker = pyg.walk(Insurance_df)

#============ matrice de correlation
corr_matrix = Insurance_df.drop(['sex', 'smoker', 'region'], axis=1).corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()
   
"""
Problematique : construire un modèle prédictif pour prédire les frais de santé afin de mieux adapter le coût de l'assurance.
"""

#============ regroupement des variables par type
numeric_feats = ["age", "bmi", "children"] # variables numériques
categ_feats = ["sex", "smoker", "region"] # variables catégorielles
target = "expenses"

#============ dummy variables
df_fe = pd.get_dummies(Insurance_df, columns=categ_feats, drop_first=True)
df_fe.head()

#============ nouvelle matrice de correlation après dummy
corr_matrix = df_fe.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

#============ variables explicatives
x = df_fe.drop("expenses", axis = 1).to_numpy()
x.shape

#============ variable à expliquer
y = df_fe["expenses"].to_numpy()
y.shape

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)

#============ MODELE 1 - entrainement du modèle
# instantiation
xgb_r = ensemble.GradientBoostingRegressor(n_estimators = 500, criterion="squared_error")

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

test_score = np.zeros((500,), dtype=np.float64)
for i, y_pred in enumerate(xgb_r.staged_predict(x_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(500) + 1,
    xgb_r.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(500) + 1, test_score, "r-", label="Test Set Deviance"
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
plt.yticks(pos, np.array(df_fe.columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    xgb_r, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(df_fe.columns)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()