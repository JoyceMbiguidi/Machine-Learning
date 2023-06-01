#============
# REGRESSION LINEAIRE MULTIPLE
# Objectif : expliquer et prédire les valeurs de plusieurs features
#============

#============ description des données
"""
The data was extracted from the 1974 Motor Trend US magazine, 
and comprises fuel consumption and 10 aspects of automobile design 
and performance for 32 automobiles (1973-74 models).

mpg : Miles/(US) gallon
cyl : Number of cylinders
disp : Displacement (cu.in.)
hp : Gross horsepower
drat : Rear axle ratio
wt : Weight (lb/1000)
qsec : 1/4 mile time
vs : V/S
am : Transmission (0 = automatic, 1 = manual)
gear : Number of forward gears
carb : Number of carburetors
"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import numpy as np

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

#============ elements graphiques
# pip install pygwalker
import pygwalker as pyg
gwalker = pyg.walk(mtcars_df)

#============ matrice de correlation
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = mtcars_df.drop(['model'], axis=1).corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

"""
Problematique : on veut savoir s'il y a une relation linéaire entre la puissance et la consommation de carburant
"""

#============ variables explicatives
x = mtcars_df.drop(['model'], axis=1).to_numpy()
x.shape

sns.histplot(data = x)

#============ variable à expliquer
y = mtcars_df['mpg'].to_numpy()
y.shape

sns.histplot(data = y)

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)

#============ MODELE 1 - entrainement du modèle
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

#============ paramètres du modèle
m = model1.coef_[0] # Le paramètre bêta1 (la pente) représente la valeur prédite de y lorsque x augmente d'une unité.
c = model1.intercept_ # bêta0 représente la valeur prédite de y lorsque x vaut 0.

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
Le modèle ne peut plus être amélioré
"""
