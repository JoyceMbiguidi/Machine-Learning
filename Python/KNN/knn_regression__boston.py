#============
# K NEAREST NEIGHBORS POUR LA REGRESSION
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


#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/Boston_dataset.csv"
raw_df = pd.read_csv(path, sep = ";")

#============ copie du dataset brut
df = raw_df
df.head()

#============ variables explicatives
x = df.drop(['medv'], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = df['medv'].to_numpy()
y.shape

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)


"""
Problematique : on veut expliquer et prédire les facteurs qui ont une influence sur le prix des maisons
"""

#============ Modele 1
from sklearn.neighbors import KNeighborsRegressor
# entrainement du modele
KNN_regression = KNeighborsRegressor(n_neighbors=5)
KNN_regression.fit(x_train, y_train)

# prédiction sur le jeu de test
y_test_predict = KNN_regression.predict(x_test)
predictions = pd.DataFrame({ 'y_test':y_test,'y_predict':y_test_predict})
predictions.head()

# évaluation du modèle sur l'ensemble de données de test
sns.scatterplot(x=y_test, y=y_test_predict, alpha=0.6, size=y_test_predict, hue=y_test_predict)
sns.regplot(x=y_test, y=y_test_predict, scatter=False, color='orange', label="Regression Line")

plt.xlabel('wage réel', fontsize=14)
plt.ylabel('wage prédit', fontsize=14)
plt.title('wage réel vs wage prédit (jeu de test)', fontsize=17)

plt.legend()

plt.show()

#============ métriques : coefficient de détermination
R2_train = KNN_regression.score(x_train, y_train)
R2_test = KNN_regression.score(x_test, y_test)
MSE_test = round(np.mean(np.square(y_test - y_test_predict)), 2)
RMSE_test = round(np.sqrt(MSE_test), 2)


print("-----------------------------------------")
print('MSE_test  = {}'.format(MSE_test))
print('RMSE_test  = {}'.format(RMSE_test))
print('R2_train = {}'.format(R2_train))
print('R2_test = {}'.format(R2_test))
print("-----------------------------------------")


#============ validation croisée : objectif, trouver l'optimum K
from sklearn.model_selection import cross_val_score
NMSE = cross_val_score(estimator = KNN_regression, X = x_train, y = y_train, cv = 5)
MSE_CV = round(np.mean(-NMSE),4)
MSE_CV


#============ choix du K
RMSE_CV=[]
RMSE_test = []

k=40

for i in range(1,k):
    KNN_i = KNeighborsRegressor(n_neighbors=i)
    KNN_i.fit(x_train, y_train)
    RMSE_i = np.sqrt(np.mean(-1*cross_val_score(estimator = KNN_i, X = x_train, y = y_train, cv = 10)))
    RMSE_CV.append(RMSE_i)
    
    RMSE_test.append(np.sqrt(np.mean(np.square(y_test - KNN_i.predict(x_test)))))
    
optimal_k = pd.DataFrame({'RMSE_CV': np.round(RMSE_CV,2), 'RMSE_test':np.round(RMSE_test,2)}, index=range(1,k))


optimal_k.head(10)

np.argmin(optimal_k['RMSE_CV'])


#============ dataviz
plt.figure(figsize=(10,5))
sns.lineplot(data=optimal_k)
plt.title('Validation croisée RMSE VS K')
plt.xlabel('K')
plt.ylabel('RMSE')
plt.show()
