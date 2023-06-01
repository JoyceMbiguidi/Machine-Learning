#============
# REGRESSION LINEAIRE SIMPLE
# Objectif : expliquer et prédire les valeurs d'une feature
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
Problematique : on veut savoir si le prix des maisons (valeur médiane) dépend du nombre moyen de chambres
"""

#============ variable explicative
x = df['rm'].to_numpy()
x.shape

#============ variable à expliquer
y = df['medv'].to_numpy()
y.shape

#============ relation lineaire entre rm et medv
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x = x, y = y, mode = "markers"))
fig.update_layout(
        title="RM vs MEDV",
        xaxis_title="RM",
        yaxis_title="MEDV")
fig.show()

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire

#============ distribution des données d'entrainement et de test
fig = go.Figure()

fig.add_trace(go.Scatter(x = x_train, y = y_train, mode = 'markers', name = 'Train'))

fig.add_trace(go.Scatter(x = x_test, y = y_test, mode = 'markers', name = 'Test'))
fig.update_layout(
        title="Distribution des données d'entrainement et de test",
        xaxis_title="RM",
        yaxis_title="MEDV")
fig.show()

#============ entrainement du modèle
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))

#============ paramètres du modèle
m = model.coef_[0] # Le paramètre bêta1 (la pente) représente la valeur prédite de y lorsque x augmente d'une unité.
c = model.intercept_ # bêta0 représente la valeur prédite de y lorsque x vaut 0.

#============ droite de regression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)
fig = go.Figure()

fig.add_trace(go.Scatter(x = x_train, y = y_train, mode = 'markers', name = 'Train'))
fig.add_trace(go.Scatter(x = x_train, y = m*x_train + c, name = "best fit" ))

fig.update_layout(
        title="Line of Best Fit Training Data",
        xaxis_title="RM",
        yaxis_title="MEDV")
fig.show()

#============ prédictions sur le jeu d'entrainement
y_predict_train = model.predict(x_train.reshape(-1,1))

#============ on affiche les valeurs prédites de y
pd.DataFrame(y_predict_train.reshape(-1,1), columns=['y_predicted']).head()

#============ On affiche les valeurs réelles de y
pd.DataFrame(y_train.reshape(-1,1), columns=['y_real']).head()

#============ évaluation du modèle sur le jeu d'entrainement
# erreur quadratique moyenne (MSE)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, y_predict_train)
MSE_train

# racine carrée de l'erreur quadratique moyenne (RMSE)
RMSE_train = mean_squared_error(y_train, y_predict_train, squared = False)
RMSE_train

# coefficient d'ajustement (R²)
from sklearn.metrics import r2_score
R_squared_train = r2_score(y_train, y_predict_train)
R_squared_train

#============ évaluation du modèle sur le jeu de test
x_test = x_test.reshape(-1,1)
y_predict_test = model.predict(x_test)

#============ on affiche les valeurs initiales de y_test
pd.DataFrame(y_test.reshape(-1,1), columns=['y_real']).head()

#============ on affiche les valeurs prédites de y_test
pd.DataFrame(y_predict_test.reshape(-1,1), columns=['y_predicted']).head()
"""
Rien que sur les 5 premières valeurs, on remarque déjà des écarts significatifs.
Il semblerait que notre modèle soit moins performant dans ses capacités prédictives, c'est normal quand on pense au R² de 50%
"""

#============ ce RMSE est légèrement plus élevé que le premier
RMSE_test = mean_squared_error(y_test, y_predict_test, squared = False)
RMSE_test

#============ coefficient d'ajustement (R²)
R_squared_test = r2_score(y_test, y_predict_test)
R_squared_test
"""
ce R² est très bas par rapport au premier. C'est la preuve qu'il existerait d'autres variables explicatives, en plus de celle utilisée
qu'on pourrait rajouter au modèle afin de mieux expliquer la variable dépendante Y.
"""

#============ un mot sur le R²
"""
- si R² = 1, on a trouvé le meilleur prédicteur possible avec un MSE = 0.
- si R² > 0, tout dépendra de sa valeur. Mais cela indique que le modèle a fait un certain nombre d'erreurs (soit peu, soit beaucoup).
- si R² < 0, signifie que les prédictions sont moins bonnes que si l'on prédisait systématiquement la valeur moyenne. Overfitting.
"""

#============ controle d'overfitting en ML
"""
permet de comparer la performance du modèle sur les jeux train et test.
Si les performances de train sont largement supérieures à celles de test, alors on a overfitté.
si l'écart est faible, alors il n'y a pas eu overfitting.
"""
