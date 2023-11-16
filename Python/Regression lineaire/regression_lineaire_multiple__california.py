#============
# REGRESSION LINEAIRE MULTIPLE
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
import seaborn as sns
import matplotlib.pyplot as plt
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
df = pd.concat([df['Price'], scaledX_df], axis = 1) # on merge les deux DF
df.head()

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(scaledX_df, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire
x_train.shape
y_train.shape
x_test.shape
y_test.shape

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
coefs = pd.DataFrame(model1.coef_, columns=['Coefficients'], index = pd.DataFrame(df).drop(['Price'], axis=1).columns)
print(coefs.sort_values)

"""
comment peut-on améliorer ce modèle ?
"""
