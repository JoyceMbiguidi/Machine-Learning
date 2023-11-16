#============
# REGRESSION LINEAIRE MULTIPLE
# Objectif : expliquer et prédire les valeurs de plusieurs features
#============

#============ description des données
"""
	Le jeu de données insurance.csv contient des informations concernant des assurés et leurs frais de santé 
	(colonne expenses). L'objectif est de construire un modèle prédictif (regression linéaire multiple) 
	pour prédire ces frais pour mieux adapter le coût de l'assurance.


age: age of primary beneficiary
sex: insurance contractor gender, female, male
bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
    objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
children: Number of children covered by health insurance / Number of dependents
smoker: Smoking
region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
charges: Individual medical costs billed by health insurance
"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import numpy as np

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/insurance.csv"
raw_df = pd.read_csv(path, sep = ",")

#============ copie du dataset brut
Insurance_df = raw_df
Insurance_df.head()

#============ vérification des types
Insurance_df.info()

#============ afficher la dimension
print(Insurance_df.shape)

#============ vérification des valeurs manquantes
print(Insurance_df.isnull().sum())

#============ elements graphiques
# pip install pygwalker
import pygwalker as pyg
gwalker = pyg.walk(Insurance_df)

#============ matrice de correlation
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = Insurance_df.drop(['sex', 'smoker', 'region'], axis=1).corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()
   
"""
Problematique : construire un modèle prédictif pour prédire les frais de santé afin de mieux adapter le coût de l'assurance.
"""

#============ regroupement des variables par type
numeric_feats = ["age", "bmi", "children"] # variables numériques
categ_feats = ["sex", "smoker", "region"] # variables catégorielles
target = "expenses"

#============ dummy variables
df_fe = pd.get_dummies(Insurance_df, columns=categ_feats, drop_first=True)
df_fe.head()

#============ nouvelle matrice de correlation après dummy
corr_matrix = df_fe.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

#============ variables explicatives
x = df_fe.drop("expenses", axis = 1).to_numpy()
x.shape

#============ variable à expliquer
y = df_fe["expenses"].to_numpy()
y.shape

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
R au carré :

Le R² (coefficient de détermination) mesure la variation expliquée par un modèle 
de régression linéaire. Plus le R² est proche de 1, meilleur est la qualité de la prédiction.
"""

"""
R au carré ajusté

Pour un modèle de régression multiple, le R² augmente ou reste le même lorsque nous ajoutons 
de nouveaux prédicteurs au modèle, même si les prédicteurs nouvellement ajoutés sont indépendants 
de la variable cible et n'ajoutent aucune valeur à la puissance de prédiction du modèle. 

Par contre, le R-carré ajusté élimine cet inconvénient du R-carré. 
Il n'augmente que si le prédicteur nouvellement ajouté améliore la puissance de prédiction du modèle. 
L'ajout de prédicteurs indépendants et non pertinents à un modèle de régression entraîne 
une diminution du R-carré ajusté.
"""

#============ variables d'impacts
# on visualise les coefficients des features
coefs = pd.DataFrame(model1.coef_, columns=['Coefficients'], index = pd.DataFrame(df_fe).drop(['expenses'], axis=1).columns)

coefs.plot(kind='barh', figsize=(9, 5))
plt.title('Regression linéaire : model 1')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)

coefs.columns

