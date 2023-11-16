#============
# SUPPORT VECTOR MACHINE POUR LA REGRESSION
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

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
corr_matrix = df.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

"""
Problematique : on veut savoir si le prix des maisons (valeur médiane) dépend du nombre moyen de chambres
"""

#============ scaling values
scale = StandardScaler()

numeric_values_scaled = scale.fit_transform(df.drop(['medv'], axis=1))
numeric_values_scaled # on obtient un array qu'il faut convertir en DataFrame
numeric_values_scaled_df = pd.DataFrame(numeric_values_scaled, columns = ['crim', 'zn', 'indus', 'chas', 
                                                                          'nox', 'rm', 'age', 'dis', 'rad', 'tax',
                                                                          'ptratio', 'b', 'lstat'])
numeric_values_scaled_df.head()


#============ on rassemble les données
df = pd.concat([df['medv'], numeric_values_scaled_df], axis=1)

df.head()


#============ variables explicatives et à expliquer
x = df.drop(['medv'], axis=1).to_numpy() # feature // variable(s) indépendante(s) // variable(s) explicative(s)
y = df['medv'].to_numpy() # target, i.e. la variable cible // variable dépendante // variable à expliquer


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

plt.xlabel('medv réel', fontsize=14)
plt.ylabel('medv prédit', fontsize=14)
plt.title('medv réel vs medv prédit (jeu de test)', fontsize=17)
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

plt.xlabel('mdev réel', fontsize=14)
plt.ylabel('medv prédit', fontsize=14)
plt.title('medv réel vs medv prédit (jeu de test)', fontsize=17)
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

























