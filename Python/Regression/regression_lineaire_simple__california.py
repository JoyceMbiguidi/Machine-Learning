#============
# REGRESSION LINEAIRE SIMPLE
# Objectif : expliquer et prédire les valeurs d'une feature
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
df = pd.DataFrame(raw_df.data, columns = raw_df.feature_names) #.data signifie variables explicatives
df.head()

#============ variable à expliquer
df['Price'] = raw_df.target # targetnumpy array of shape (20640,).
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
Le coef de corrélation varie de -1 à 1.
Proche de 1 = variation dans le même sens. Si a augmente, alors b augmente.
Proche de -1 = variation dans le sens opposé. Si a augmente, alors b baisse.
coef = 0, alors absence de corrélation. La variation de a n'a aucune incidence sur b.

Fort lien entre Price et MedInc. Plus le prix du logement est élevé, plus le revenu est élevé.
Fort lien entre AveBedrms et AveRooms. Plus il y a de pièces dans un logement, plus il y a de forte chances qu'il y ait un nombre élevé de chambres.
"""

"""
Problematique : on veut expliquer le prix par le revenu médian
"""

x = df['MedInc'].to_numpy() # feature // variable(s) indépendante(s) // variable(s) explicative(s)
y = df['Price'].to_numpy() # target, i.e. la variable cible // variable dépendante // variable à expliquer

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire
x_train.shape
y_train.shape
x_test.shape
y_test.shape

#============ entrainement du modèle sur le jeu d'entrainement
from sklearn.linear_model import LinearRegression

model = LinearRegression()
x_train = x_train.reshape(-1,1) # une ligne, une colonne
model.fit(x_train, y_train)

#============ prédictions sur le jeu d'entrainement
y_predict_train = model.predict(x_train)
y_predict_train.reshape(-1, 1).round(2) # affiche les valeurs prédites Y chapeau
y_train.reshape(-1, 1).round(2) # affiche les valeurs réelles de Y

#============ évaluation du modèle sur le jeu d'entrainement
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

"""
moyenne des carrés des erreurs
Cette métrique est très sensible aux outliers (vraies valeurs de Y).
La prédiction sera donc souvent très éloignée des valeurs aberrantes.
"""
mse_train = mean_squared_error(y_train, y_predict_train)
mse_train

"""
La racine carrée de la MSE
Contrairement à la MSE, la RMSE s’exprime dans la même unité que la variable à prédire et est par conséquent plus facile à interpréter. 
Ces métriques quantifient les erreurs réalisées par le modèle. Plus elles sont élevées, moins le modèle est performant. 
Cette métrique est très sensible aux outliers (vraies valeurs de Y).
La prédiction sera donc souvent très éloignée des valeurs aberrantes.
"""
rmse_train = mean_squared_error(y_train, y_predict_train, squared = False)
rmse_train

r2_train = r2_score(y_train, y_predict_train)
r2_train

#============ paramètres du modèle
m = model.coef_[0] # Le paramètre bêta1 (la pente) représente la valeur prédite de y lorsque x augmente d'une unité.
c = model.intercept_ # bêta0 représente la valeur prédite de y lorsque x vaut 0.

"""
D'abord regarder le signe du coefficient : (+) ou (-) ?
Si positif, alors il influence Y de façon à l'augmenter en moyenne.
Si négatif, alors il influence Y de façon à le baisser en moyenne.

Interprétation : à chaque fois que MedInc augmente d'une unité (exprimée dans l'unité de x), 
Y augmente en moyenne de +0,41 (exprimée dans l'unité de Y).
"""

#============ prédictions sur le jeu de test
y_predict_test = model.predict(x_test.reshape(-1,1))

#============ évaluation du modèle sur le jeu de test
rmse_test = mean_squared_error(y_test, y_predict_test, squared = False)
rmse_test
r2_test = r2_score(y_test, y_predict_test)
r2_test

"""
Comment peut-on améliorer notre modèle ?
    # trouver d'autres features à intégrer au modèle
    # supprimer une ou plusieurs features du modèle
    # améliorer la qualité des données (fraîcheur, complétude, filtrer, ..., quantité)
    # transformation des variables (dummy variables (ou dichotomie), log(x), scale(X), ...)
"""
