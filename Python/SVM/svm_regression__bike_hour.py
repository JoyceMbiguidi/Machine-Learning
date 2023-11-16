#============
# SUPPORT VECTOR MACHINE POUR LA REGRESSION
# Objectif : expliquer et prédire les valeurs d'une feature
#============

#============ description des données
"""
# Décomptes de partage de vélos agrégés sur une base horaire. Enregistrement : 17379 heures #

Le processus de location de vélos en libre-service est fortement corrélé aux paramètres environnementaux et saisonniers.
Par exemple, les conditions météorologiques, les précipitations, le jour de la semaine, la saison, 
l'heure de la journée, etc. peuvent affecter les comportements de location. 

	- instant: record index
	- dteday : date
	- season : season (1:springer, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2011, 1:2012)
	- mnth : month ( 1 to 12)
	- hr : hour (0 to 23)
	- holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit : 
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
	- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
	- hum: Normalized humidity. The values are divided to 100 (max)
	- windspeed: Normalized wind speed. The values are divided to 67 (max)
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered
"""


#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/bike_hour.csv"
raw_df = pd.read_csv(path, sep = ";", decimal = ".")

#============ copie du dataset brut
bike_df = raw_df
bike_df.head()

#============ vérification des types
bike_df.dtypes

#============ afficher la dimension
print(bike_df.shape)

#============ data wrangling
bike_df.isna().sum()  

# changement de type de variables
categorical = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday','weathersit', 'workingday']

for col in categorical: 
    bike_df[col] = bike_df[col].astype("category")

#============ vérification des types
bike_df.dtypes

#============ suppression d'une colonne de peu d'intérêt
bike_df.drop(['dteday', 'instant'],axis=1, inplace=True)


#============ regroupement des variables par type
numeric_feat = [
    'cnt',
    'registered',
    'casual',
    'windspeed',
    'hum',
    'atemp',
    'temp'
]


categ_feat = [
    'holiday',
	'hr',
	'mnth',
	'season',
	'weathersit',
	'weekday',
	'workingday',
	'yr'
]


#============ Dataviz

# matrice de corrélation
corr_matrix = bike_df[bike_df.columns.difference(categ_feat)].corr().round(2)
print(corr_matrix)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

# pair plot
sns.set_style('whitegrid')
sns.pairplot(bike_df[bike_df.columns.difference(categorical)], corner=True)


#============ dummy variables
bike_df = pd.get_dummies(bike_df, drop_first=True)
bike_df.head()


#============ standardisation des variables continues

"""
Il n'est pas préférable de standardiser Y.
Certains Data Scientists le font, et cela change la façon d'interpréter les effets du coefficients de la variable explicative sur Y.

Cas 1 : Y et X restent inchangés : on maintient l'interprétation des coefs abordé dans le module "Régression Linéaire".
Cas 2 : Y reste inchangé et X est standardisé : c'est la forme de modélisation la plus répandue. 
Cas 3 : Y est standardisé et X reste inchangé : jamais expérimenté et je doute de la pertinence d'une telle modélisation.


Explication du cas 1 :
    Ici, lorsque X augmente ou baisse d'une unité supplémentaire, Y augmente ou baisse en moyenne selon la valeur
    du coefficient bêta de X, ce dernier exprimé dans l'unité de mesure de Y.
    
Explication du cas 2 :
    Ici, on raisonne en termes d'écart-type.
    Lorsque la variable explicative est standardisée, le coefficient de régression représente le changement standardisé 
    de la variable dépendante associé à un changement d'une unité standard (c'est-à-dire un écart-type) dans la variable explicative. 
    Cela signifie que pour une augmentation d'une unité standard de la variable explicative, la variable dépendante change en termes d'écart-type.
    
    Impact sur l'échelle d'origine : Pour obtenir l'impact de la variable explicative sur l'échelle d'origine de la 
variable dépendante, vous pouvez multiplier le coefficient de régression standardisé par l'écart-type de la 
variable dépendante. Cela vous donne l'effet de la variable explicative sur la variable dépendante en unités d'origine.
"""


scale = StandardScaler()
bike_df_scaled = scale.fit_transform(bike_df[['registered', 'casual', 'windspeed', 'hum', 'atemp', 'temp']]) # je retire mon Y et les dummy
bike_df_scaled = pd.DataFrame(bike_df_scaled, columns = bike_df[['registered', 'casual', 'windspeed', 'hum', 'atemp', 'temp']].columns)

bike_df_scaled = pd.concat([bike_df_scaled, bike_df['cnt'], bike_df.filter(regex='_', axis=1)], axis=1)


"""
Problematique : Quels sont les déterminants de la location de vélos ?
"""

#============ variable explicative
x = bike_df_scaled.drop('cnt', axis = 1)
x.shape

#============ variable à expliquer
y = bike_df_scaled['cnt']
y.shape

sns.histplot(data = y)


#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire


#============ Modele 1
# ajustement du modèle de régression SVM sur l'ensemble d'entrainement
from sklearn.svm import SVR

SVM_regression = SVR()
SVM_regression.fit(x_train, y_train)
SVR()

# prédiction sur le jeu de test
y_predict = SVM_regression.predict(x_test)
predictions = pd.DataFrame({ 'y_test':y_test,'y_predict':y_predict})
predictions.head()

# évaluation du modèle sur l'ensemble de données de test
sns.scatterplot(x=y_test, y=y_predict, alpha=0.6)
sns.lineplot(x=y_test, y=y_test, color='red', label='Ligne de régression')

plt.xlabel('cnt réel', fontsize=14)
plt.ylabel('cnt prédit', fontsize=14)
plt.title('cnt réel vs cnt prédit (jeu de test)', fontsize=17)
plt.show()

'''
on serait tenté d'utiliser un kernel POLY et comparer les R²' : SVM_regression = SVR(kernel = 'poly')
'''


#============ métrique : coefficient de détermination
SVM_regression.score(x_test, y_test)
MSE_test = round(np.mean(np.square(y_test - y_predict)), 2)
RMSE_test = round(np.sqrt(MSE_test), 2)
R2_test = r2_score(y_test, y_predict)

print("-----------------------------------------")
print('MSE_test  = {}'.format(MSE_test))
print('RMSE_test  = {}'.format(RMSE_test))
print('R2_test = {}'.format(R2_test))
print("-----------------------------------------")


#============ Modele 2
#============ Tuning des hyper paramètres avec Gridsearch
'''
Rappel sur quelques paramètres :
   - C représente le coût d’une mauvaise classification. 
    Un grand C signifie que vous pénalisez les erreurs de manière plus stricte, donc la marge sera plus étroite, 
    c'est-à-dire un surajustement (petit biais, grande variance).
   
   - gamma est le paramètre libre dans la fonction de base radiale (rbf). 
   Intuitivement, le paramètre gamma (inverse de la variance) définit jusqu'où va l'influence d'un seul exemple d'entraînement, 
   des valeurs faibles signifiant « loin » et des valeurs élevées signifiant « proche ».
'''

my_param_grid = {'C': [1,10,100], 'gamma': [1,0.1,0.01], 'kernel': ['rbf']}

'''
nous avons défini une grille d'hyper paramètres pour un SVM avec un noyau (kernel) de type 'rbf' (radial basis function), 
une fonction de base radiale).
    
    - "C" : C'est l'hyperparamètre de régularisation. Il contrôle la marge d'erreur du SVM. 
    Trois valeurs différentes sont testées : 1, 10 et 100. Des valeurs plus élevées de C permettent au SVM 
    d'ajuster davantage aux données d'entraînement, mais cela peut également entraîner un surajustement (overfitting).
    
    - "gamma" : C'est un autre hyperparamètre important qui contrôle la forme de la fonction de noyau rbf. 
    Trois valeurs différentes sont testées : 1, 0.1 et 0.01. 
    Une valeur plus élevée de gamma signifie que le modèle attribue plus d'importance aux points de données proches.
    
    - "kernel" : Ici seul le noyau 'rbf' qui est spécifié. 
    Le noyau rbf est largement utilisé pour les SVM et est souvent la meilleure option par défaut pour de nombreuses tâches.
'''

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=SVR(), param_grid= my_param_grid, refit = True, verbose=2, cv=5, n_jobs=-1)

# ajustement du modele avec les hyper parametres
grid.fit(x_train, y_train)


# meilleurs paramètres gridsearch
grid.best_params_
grid.best_estimator_

y_predict_optimized = grid.predict(x_test)

predictions['y_predict_optimized'] = y_predict_optimized
predictions.head()


#============ visualisation graphique après optimisation
sns.scatterplot(x=y_test, y=y_predict_optimized, alpha=0.6)
sns.lineplot(x=y_test, y=y_test, color='red', label='Ligne de régression')

plt.xlabel('cnt réel', fontsize=14)
plt.ylabel('cnt prédit', fontsize=14)
plt.title('cnt réel vs cnt prédit (jeu de test)', fontsize=17)
plt.show()


#============ métrique : coefficient de détermination
grid.score(x_test, y_test)
MSE_test_model2_opt = round(np.mean(np.square(y_test - y_predict_optimized)), 2)
RMSE_test_model2_opt = round(np.sqrt(MSE_test_model2_opt), 2)
R2_test_model2_opt = r2_score(y_test, y_predict_optimized)

print("-----------------------------------------")
print('MSE_test_model2_opt  = {}'.format(MSE_test_model2_opt))
print('RMSE_test_model2_opt  = {}'.format(RMSE_test_model2_opt))
print('R2_test_model2_opt = {}'.format(R2_test_model2_opt))
print("-----------------------------------------")


"""
Peut-on améliorer ce modèle ? Testons le avec un kernel de type polynomial...
"""

#============ Modele 3
model3_param_grid = {'C': [1,10,100], 'gamma': [1,0.1,0.01], 'kernel': ['poly']}

model3_grid = GridSearchCV(estimator=SVR(), param_grid= my_param_grid, refit = True, verbose=2, cv=5 )

# ajustement du modele avec les hyper parametres
model3_grid.fit(x_train, y_train)


# meilleurs paramètres gridsearch
model3_grid.best_params_
model3_grid.best_estimator_

model3_y_predict_optimized = model3_grid.predict(x_test)

predictions['model3_y_predict_optimized'] = model3_y_predict_optimized
predictions.head()


#============ visualisation graphique après optimisation
sns.scatterplot(x=y_test, y=model3_y_predict_optimized, alpha=0.6)
sns.lineplot(y_test, y_test)

plt.xlabel('Actual count', fontsize=14)
plt.ylabel('Prediced  count', fontsize=14)
plt.title('Actual vs optimized predicted count (test set)', fontsize=17)
plt.show()


#============ métrique : coefficient de détermination
grid.score(x_test, y_test)
MSE_test_model3_opt = round(np.mean(np.square(y_test - y_predict_optimized)), 2)
RMSE_test_model3_opt = round(np.sqrt(MSE_test_model3_opt), 2)
R2_test_model3_opt = r2_score(y_test, y_predict_optimized)

print("-----------------------------------------")
print('MSE_test_model3_opt  = {}'.format(MSE_test_model3_opt))
print('RMSE_tes_model3t_opt  = {}'.format(RMSE_test_model3_opt))
print('R2_test_model3_opt = {}'.format(R2_test_model3_opt))
print("-----------------------------------------")



"""
Métriques de SVM pour la régression :

    - Erreur quadratique moyenne (Mean Squared Error, MSE) : Cette métrique mesure la moyenne des carrés des erreurs 
    entre les valeurs prédites et les valeurs réelles. Plus le MSE est bas, meilleure est la performance du modèle.

    - Erreur absolue moyenne (Mean Absolute Error, MAE) : Le MAE mesure la moyenne des valeurs absolues des erreurs. 
    Il est plus robuste aux valeurs aberrantes que le MSE.

    - Coefficient de détermination (R²) : Le R² mesure la proportion de la variance des valeurs de sortie qui est 
    expliquée par le modèle. Il indique à quel point le modèle est adapté aux données.

    - Erreur quadratique moyenne logarithmique (Mean Squared Logarithmic Error, MSLE) : Cette métrique est utile 
    lorsque les valeurs de sortie ont une échelle logarithmique.

    - Erreur absolue moyenne logarithmique (Mean Absolute Logarithmic Error, MALE) : Cette métrique est une alternative au MSLE.
"""


