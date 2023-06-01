#============
# REGRESSION LINEAIRE MULTIPLE
# Objectif : expliquer et prédire les valeurs de plusieurs features
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

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/ozone.csv"
raw_df = pd.read_csv(path, sep = ";", decimal = ",")

#============ copie du dataset brut
ozone_df = raw_df
ozone_df.head()

#============ vérification des types
ozone_df.dtypes

#============ afficher la dimension
print(ozone_df.shape)

#============ elements graphiques
# pip install pygwalker
import pygwalker as pyg
gwalker = pyg.walk(ozone_df)

#============ matrice de correlation
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = ozone_df.drop(['vent', 'pluie'], axis=1).corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

"""
Problematique : On veut déterminer les variables qui ont un impact sur la concentration journalière en ozone
"""

#============ variables explicatives
x = ozone_df.drop(['obs', 'maxO3', 'vent', 'pluie'], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = ozone_df['maxO3'].to_numpy()
y.shape

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire

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

#============ variables d'impacts
# on visualise les coefficients des features
coefs = pd.DataFrame(model1.coef_, columns=['Coefficients'], index = pd.DataFrame(ozone_df).drop(['obs', 'maxO3', 'vent', 'pluie'], axis=1).columns)

coefs.plot(kind='barh', figsize=(9, 5))
plt.title('Regression linéaire : model 1')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)


"""Peut-on améliorer le R² ? Si oui, de quelle(s) façon(s) ?"""

#============ MODELE 2 - entrainement du modèle avec stepwise
# pip install stepwise-regression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from stepwise_regression import step_reg

#============ statsmodels avec toutes les features du jeu d'entrainement
# on passe de numpy array à un DataFrame
x_train_df = pd.DataFrame(x_train, columns=['T9', 'T12', 'T15', 'Ne9', 'Ne12', 'Ne15', 'Vx9', 'Vx12', 'Vx15', 'maxO3v'])
y_train_df = pd.DataFrame(y_train, columns=['maxO3'])

model_smf = smf.ols(formula='y_train_df ~ T9 + T12 + T15 + Ne9 + Ne12 + Ne15 + Vx9 + Vx12 + Vx15 + maxO3v', data = x_train_df).fit()

#============ resultats de statsmodels
print(model_smf.summary())

#============ model avec stepwise
backselect = step_reg.backward_regression(x_train_df, y_train_df, 0.05, verbose=True) # 0.05 est la valeur seuil p-value
backselect

#============ nouveau modèle après stepwise
model_smf = smf.ols(formula='y_train_df ~ T12 + Ne9 + maxO3v', data = x_train_df).fit()
print(model_smf.summary())

#============ statsmodels avec toutes les features du jeu de test
# on passe de numpy array à un DataFrame
x_test_df = pd.DataFrame(x_test, columns=['T9', 'T12', 'T15', 'Ne9', 'Ne12', 'Ne15', 'Vx9', 'Vx12', 'Vx15', 'maxO3v'])
y_test_df = pd.DataFrame(y_test, columns=['maxO3'])

model_smf = smf.ols(formula='y_test_df ~ T9 + T12 + T15 + Ne9 + Ne12 + Ne15 + Vx9 + Vx12 + Vx15 + maxO3v', data = x_test_df).fit()

#============ resultats de statsmodels
print(model_smf.summary())

#============ model avec stepwise
backselect = step_reg.backward_regression(x_test_df, y_test_df, 0.05, verbose=True) # 0.05 est la valeur seuil p-value
backselect

#============ nouveau modèle après stepwise
model_smf = smf.ols(formula='y_test_df ~ T12 + Ne9 + maxO3v', data = x_test_df).fit()
print(model_smf.summary())

