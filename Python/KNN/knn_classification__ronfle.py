#============
# K NEAREST NEIGHBORS POUR LA CLASSIFICATION
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle binaire
#============

#============ description des données
"""
On fait une étude sur le ronflement. Nous avons des caractéristiques physiologiques et d'habitudes de consommation de quelques patients.

caractéristiques physiologiques : taille, poids, age, sexe...
habitudes de consommation : alcool, fumeur...
Problématique : On veut savoir ce qui provoque le ronflement autrement dit, quelles sont les facteurs responsables du ronflement ?
"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/ronfle.txt"
raw_df = pd.read_csv(path, sep = "\t")

#============ copie du dataset brut
ronfle_df = raw_df
ronfle_df.head()

#============ vérification des types
ronfle_df.dtypes

#============ afficher la dimension
print(ronfle_df.shape)

#============ recherche des valeurs manquantes
ronfle_df.isnull().sum()

"""
Problematique : quels sont les facteurs responsables du ronflement ?
"""

#============ recodage des variables
ronfle_df.replace(('N', 'O'), (0, 1), inplace=True) # recodage des variables
ronfle_df.replace(('F', 'H'), (0, 1), inplace=True) # recodage des variables
ronfle_df.head()

#============ variables explicatives
x = ronfle_df.drop(['ronfle'], axis = 1)
x.shape

#============ variable à expliquer
y = ronfle_df['ronfle']

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)

#============ scaling values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_scaled = sc.transform(x_train)
x_test_scaled = sc.transform(x_test)

#============ Modele 1
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(x_train_scaled, y_train)
y_pred = KNN.predict(x_test_scaled)

from sklearn.metrics import accuracy_score
print("Test accuracy: ", accuracy_score(y_test, y_pred))


#============ lets choose the right number of K using grid search
k_values = list(range(1, 31))

# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Define the hyperparameter grid
param_grid = {'n_neighbors': k_values}

# Initialize the GridSearchCV object
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(knn, param_grid=param_grid, scoring='accuracy', cv=10)

# Fit the grid search to the training data
grid_search.fit(x_train_scaled, y_train)

# Print the results
print("Best value of K: ", grid_search.best_params_)
print("Mean CV accuracy of best K-value: {:.3f}".format(grid_search.best_score_))


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot([params['n_neighbors'] for params in grid_search.cv_results_['params']], 
         grid_search.cv_results_['mean_test_score'], 'bo-')
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean CV Accuracy')
plt.title('KNN Grid Search Results')
plt.show()

# predictions be like
y_pred = KNN.predict(x_test_scaled)
