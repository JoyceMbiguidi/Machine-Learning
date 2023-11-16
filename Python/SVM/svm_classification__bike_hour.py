#============
# SUPPORT VECTOR MACHINE POUR LA CLASSIFICATION
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle binaire
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


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


"""
Problematique : Le manager du magasin veut optimiser la répartition des vélos dans les autres stations de la ville. 
On veut donc identifier les moments où 500 vélos sont loués en une heure, pour approvisionner d'autres stations avec 
des vélos supplémentaires.
"""

#============ identification du point critique
overload = np.where(bike_df['cnt']>500,1,0)

pd.crosstab(overload,overload, normalize='all')*100
# les modalités sont déséquilibrées

#============ ajout de cette information dans le tableau initial
bike_df['overload'] = overload
bike_df.drop('cnt', axis=1, inplace=True)
bike_df.columns

#============ suppression d'une colonne de peu d'intérêt
bike_df.drop(['dteday', 'instant'],axis=1, inplace=True)

# changement de type de variables
categorical = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday','weathersit', 'workingday']

for col in categorical: 
    bike_df[col] = bike_df[col].astype("category")


#============ Dataviz

# matrice de corrélation
corr_matrix = bike_df[bike_df.columns.difference(categorical)].corr().round(2)
print(corr_matrix)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

# pair plot
sns.set_style('whitegrid')
sns.pairplot(bike_df[bike_df.columns.difference(categorical)], hue='overload')


#============ dummy variables
bike_df = pd.get_dummies(bike_df, drop_first=True)
bike_df.head()

#============ standardisation des variables continues
scale = StandardScaler()
bike_df_scaled = scale.fit_transform(bike_df[['registered', 'casual', 'windspeed', 'hum', 'atemp', 'temp']]) # je retire mon Y et les dummy
bike_df_scaled = pd.DataFrame(bike_df_scaled, columns = bike_df[['registered', 'casual', 'windspeed', 'hum', 'atemp', 'temp']].columns)

bike_df_scaled = pd.concat([bike_df_scaled, bike_df['overload'], bike_df.filter(regex='_', axis=1)], axis=1)
bike_df_scaled.head()


#============ variable explicative
x = bike_df_scaled.drop('overload', axis = 1)
x.shape

#============ variable à expliquer
y = bike_df_scaled['overload']
y.shape

sns.histplot(data = y)



#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire


#============ Modele 1
from sklearn.svm import SVC
# ajustement du modele de classification sur le jeu d'entrainement
SVM_classification = SVC()
SVM_classification.fit(x_train, y_train)


# prédiction sur le jeu de test
y_predict = SVM_classification.predict(x_test)
predictions = pd.DataFrame({ 'y_test':y_test,'y_predict':y_predict})
predictions.head(15)


# évaluation du modèle sur l'ensemble de données de test
def my_SVM_report(x_train, y_train, X_test,y_test, C=1,gamma='scale' ,kernel='rbf'):
    svc= SVC(C=C, gamma=gamma, kernel=kernel)
    svc.fit(x_train, y_train)
    y_predict = svc.predict(x_test)
    
    cm = confusion_matrix(y_test, y_predict)
    accuracy = round(accuracy_score(y_test,y_predict) ,4)
    error_rate = round(1-accuracy,4)
    precision = round(precision_score(y_test,y_predict),2)
    recall = round(recall_score(y_test,y_predict),2)
    f1score = round(f1_score(y_test,y_predict),2)
    cm_labled = pd.DataFrame(cm, index=['Réel : négatif ','Réel : positif'], columns=['Prédit : negatif','Prédit : positif'])
    
    print("-----------------------------------------")
    print('Accuracy  = {}'.format(accuracy))
    print('Error_rate  = {}'.format(error_rate))
    print('Precision = {}'.format(precision))
    print('Recall    = {}'.format(recall))
    print('f1_score  = {}'.format(f1score))
    print("-----------------------------------------")
    return cm_labled


my_SVM_report(x_train, y_train, x_test, y_test, kernel='rbf')


"""
peut-on améliorer ce modèle ?
"""

#============ Tuning des hyper paramètres avec Gridsearch
#============ Modele 2

"""
Rappel sur quelques paramètres :
    - C représente le coût d’une mauvaise classification. Un grand C signifie que vous pénalisez les erreurs de 
    manière plus stricte, donc la marge sera plus étroite, c'est-à-dire un surajustement (petit biais, grande variance).
    
    - gamma est le paramètre libre dans la fonction de base radiale (rbf). Intuitivement, le paramètre gamma (inverse de la variance) 
    définit jusqu'où va l'influence d'un seul exemple d'entraînement, des valeurs faibles signifient « loin » et des valeurs élevées signifient « proche ».
"""

my_param_grid = {'C': [10,100,1000], 'gamma': ['scale',0.01,0.001], 'kernel': ['rbf']} 

from sklearn.model_selection import GridSearchCV

GridSearchCV(estimator=SVC(),param_grid= my_param_grid, refit = True, verbose=2, cv=5 )
model2_grid = GridSearchCV(estimator=SVC(),param_grid= my_param_grid, refit = True, verbose=2, cv=5, n_jobs=-1)


# ajustement du modele avec les hyper parametres
model2_grid.fit(x_train, y_train)

# meilleurs paramètres gridsearch
model2_grid.best_params_
model2_grid.best_estimator_

model2_y_predict_optimized = model2_grid.predict(x_test)

predictions['model2_y_predict_optimized'] = model2_y_predict_optimized
predictions.head(15)


#============ rapport des métriques
my_SVM_report(x_train, y_train, x_test, y_test, C=1000, gamma=0.001)


"""
Métriques de SVM pour la classification :

    - Accuracy (Précision) : C'est la mesure la plus courante pour évaluer la performance d'un modèle de classification. 
    Elle indique la proportion de prédictions correctes parmi toutes les prédictions.

    - Matrice de confusion : La matrice de confusion permet de visualiser les vrais positifs, les vrais négatifs, 
    les faux positifs et les faux négatifs. Elle est utile pour évaluer la capacité de votre modèle à discriminer entre les classes.

    - Précision : La précision mesure la proportion de vrais positifs parmi les prédictions positives. 
    Elle est particulièrement utile lorsque vous voulez minimiser les faux positifs.

    - Rappel (Sensibilité) : Le rappel mesure la proportion de vrais positifs parmi tous les exemples positifs réels. 
    Il est important lorsque vous voulez minimiser les faux négatifs.

    - F1-score : L'F1-score est une métrique qui tient compte à la fois de la précision et du rappel. Elle est utile 
    lorsque vous avez besoin d'équilibrer la précision et le rappel.

    - Courbe ROC et Aire sous la courbe (AUC) : Ces métriques sont utiles pour évaluer la capacité de discrimination 
    du modèle. L'AUC mesure la capacité du modèle à classer correctement les exemples positifs et négatifs.
"""
