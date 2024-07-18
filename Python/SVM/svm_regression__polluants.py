#============
# SUPPORT VECTOR MACHINE POUR LA REGRESSION
# Objectif : expliquer et prédire les valeurs d'une feature
#============

#============ description des données
"""
Emissions de CO2 et de polluants des véhicules commercialisés en France

    DESCRIPTION
Depuis 2001, l’ADEME acquiert tous les ans ces données auprès de l’Union Technique de l’Automobile du motocycle et 
du Cycle UTAC (en charge de l’homologation des véhicules avant leur mise en vente) en accord avec le ministère 
du développement durable.
Pour chaque véhicule les données d’origine (transmises par l’Utac) sont les suivantes :

	- les consommations de carburant
	- les émissions de dioxyde de carbone (CO2)
	- les émissions des polluants de l’air (réglementés dans le cadre de la norme Euro)
	- l’ensemble des caractéristiques techniques des véhicules (gammes, marques, modèles, n° de CNIT, type d’énergie ...)
    
    
    Types de carburants:

AC : air comprimé
EE : essence-électricité (hybride rechargeable)
EG : essence-GPL
EH : véhicule hybride non rechargeable
EL : électricité
EM : essence-gaz naturel et électricité rechargeable
EN : essence-gaz naturel
EP : essence-gaz naturel et électricité non rechargeable
EQ : essence-GPL et électricité non rechargeable

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
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/polluants.csv"
raw_df = pd.read_csv(path, sep = ";", decimal = ",", encoding = "ISO-8859-1")

#============ copie du dataset brut
polluants_df = raw_df
polluants_df.head()

#============ vérification des types
polluants_df.dtypes

#============ afficher la dimension
print(polluants_df.shape)

#============ on conserve les colonnes d'intérêts
polluants_df = polluants_df[['co2', 'cod_cbr', 'hybride', 'Carrosserie', 'gamme', 'puiss_admin_98', 'conso_urb', 
                             'conso_exurb', 'conso_mixte', 'hc', 'nox', 'hcnox']]

#============ data wrangling
polluants_df.isna().sum()  

# imputation des valeurs manquantes par la moyenne (ça se discutte pour certaines variables)
# Si écart-type est grand (proche ou supérieur à la valeur moyenne, alors j'impute par la médiane)
polluants_df['co2'].fillna(polluants_df['co2'].mean(), inplace = True)
polluants_df['conso_urb'].fillna(polluants_df['conso_urb'].mean(), inplace = True)
polluants_df['conso_exurb'].fillna(polluants_df['conso_exurb'].mean(), inplace = True)
polluants_df['conso_mixte'].fillna(polluants_df['conso_mixte'].mean(), inplace = True)
polluants_df['hcnox'].fillna(polluants_df['hcnox'].mean(), inplace = True)
polluants_df['conso_urb'].fillna(polluants_df['conso_urb'].mean(), inplace = True)
polluants_df['conso_exurb'].fillna(polluants_df['conso_exurb'].mean(), inplace = True)
polluants_df['conso_mixte'].fillna(polluants_df['conso_mixte'].mean(), inplace = True)
polluants_df['hc'].fillna(polluants_df['hc'].mean(), inplace = True)
polluants_df['nox'].fillna(polluants_df['nox'].mean(), inplace = True)
polluants_df['hcnox'].fillna(polluants_df['hcnox'].mean(), inplace = True)


#============ regroupement des variables par type
target = polluants_df[['co2']] # ma target, mon Y
numeric_values = polluants_df[['puiss_admin_98', 'conso_urb','conso_exurb', 'conso_mixte','hc','nox','hcnox']] # ensemble des valeurs quantitatives
categorical_values = polluants_df[['cod_cbr','hybride','gamme']] # ensemble des valeurs qualitatives

numeric_values.head()
categorical_values.head()


#============ Dataviz
# matrice de corrélation
corr_matrix = polluants_df[polluants_df.columns.difference(categorical_values)].corr().round(2)
print(corr_matrix)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()


#============ dummy variables
dummy = pd.get_dummies(categorical_values,
                     columns = categorical_values.columns, drop_first=True)

dummy.head()


#============ scaling values
scale = StandardScaler()

numeric_values_scaled = scale.fit_transform(numeric_values)
numeric_values_scaled # on obtient un array qu'il faut convertir en DataFrame
numeric_values_scaled_polluants_df = pd.DataFrame(numeric_values_scaled, columns = ['puiss_admin_98', 'conso_urb','conso_exurb', 'conso_mixte','hc','nox','hcnox'])
numeric_values_scaled_polluants_df.head()


#============ on rassemble les données
polluants_df = pd.concat([target, numeric_values_scaled_polluants_df, dummy], axis = 1) # on merge les deux polluants_df
polluants_df.head()


"""
Problematique : Quels sont les déterminants d'émissions de polluants ?
"""

#============ variable explicative
x = polluants_df.drop(['co2'], axis = 1)

#============ variable à expliquer
y = polluants_df['co2']

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)


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
sns.lineplot(x=y_test, y=y_test, color='red', label="Ligne d'ajustement")

plt.xlabel('co2 réel', fontsize=14)
plt.ylabel('co2 prédit', fontsize=14)
plt.title('co2 réel vs co2 prédit (jeu de test)', fontsize=17)
plt.show()


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


""" comment améliorer ce modèle ?"""


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

my_param_grid = {'C': [1], 'gamma': [1,0.1], 'kernel': ['rbf']}

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
MSE_test_opt = round(np.mean(np.square(y_test - y_predict_optimized)), 2)
RMSE_test_opt = round(np.sqrt(MSE_test_opt), 2)
R2_test_opt = r2_score(y_test, y_predict_optimized)

print("-----------------------------------------")
print('MSE_test_opt  = {}'.format(MSE_test_opt))
print('RMSE_test_opt  = {}'.format(RMSE_test_opt))
print('R2_test_opt = {}'.format(R2_test_opt))
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






















