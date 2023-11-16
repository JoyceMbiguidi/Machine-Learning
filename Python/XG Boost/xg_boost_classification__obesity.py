#============
# EXTREME GRADIENT BOOSTING POUR LA CLASSIFICATION
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle binaire
#============

#============ description des données
"""
This dataset include data for the estimation of obesity levels in individuals from the countries of Mexico, 
Peru and Colombia, based on their eating habits and physical condition. 
The data contains 17 attributes and 2111 records, the records are labeled with the class variable NObesity (Obesity Level), 
that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, 
Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 77% of the data was generated synthetically 
using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users through a web platform.

gender : Female or Male
age :	Numeric value
height : Numeric value in meters
weight : Numeric value in kilograms
family_history_with_overweight : Has a family member suffered or suffers from overweight ?  Yes or No
FAVC : Do you eat high caloric food frequently?	Yes or No
FCVC : Do you usually eat vegetables in your meals? Never, Sometimes, Always
NCP : How many main meals do you have daily? Between 1 y 2, Three, More than three
CAEC : Do you eat any food between meals? No, Sometimes, Frequently, Always
SMOKE : Yes, No
CH20 : How much water do you drink daily? Less than a liter, Between 1 and 2 L, More than 2 L
SCC : Do you monitor the calories you eat daily? Yes, No
FAF : How often do you have physical activity? I do not have, 1 or 2 days, 2 or 4 days, 4 or 5 days
TUE : How much time do you use technological devices such as cell phone, videogames, television, computer and others? 0–2 hours, 3–5 hours, More than 5 hours
CALC : how often do you drink alcohol? I do not drink, Sometimes, Frequently, Always
MTRANS : Which transportation do you usually use? Automobile, Motorbike, Bike, Public Transportation, Walking
NObeyesdad
"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import seaborn as sns
import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/obesity.csv"
raw_df = pd.read_csv(path, sep = ";")

#============ copie du dataset brut
obesity_df = raw_df
obesity_df.head()

#============ afficher la dimension
print(obesity_df.shape)

#============ vérification des types
obesity_df.dtypes

#============ recherche des valeurs manquantes
print(obesity_df.isnull().sum())

#============ statistiques descriptives
obesity_df.describe()
obesity_df.describe(include = [object]).transpose()


#============ dataviz
# import pygwalker as pyg
# from IPython.display import display, HTML
# gwalker = pyg.walk(obesity_df)
# works fine on jupiter notebook

# visualisation des variables numeriques
sns.displot(obesity_df['Age'], kde=True)
sns.displot(obesity_df['Height'], kde=True)
sns.displot(obesity_df['Weight'], kde=True)


#============ wrangling
# discretize variables
obesity_df["Age"] = pd.cut(obesity_df.Age, bins = 3, labels = ["young", "adult", "senior"])
obesity_df["Height"] = pd.cut(obesity_df.Height, bins = 3, labels = ["short", "middle", "tall"])
obesity_df["Weight"] = pd.cut(obesity_df.Weight, bins = 3, labels = ["fly weight", "welt'er weight", "heavy weight"])
obesity_df["FCVC"] = pd.cut(obesity_df.FCVC, bins = 3, labels = ['never eat vegetables', 'sometimes eat vegetables', 'always eat vegetables'])
obesity_df["NCP"] = pd.cut(obesity_df.NCP, bins = 4, labels = ['one main meals', 'two  main meals', 'three  main meals', 'more than 3 main meals'])
obesity_df["CH2O"] = pd.cut(obesity_df.CH2O, bins = 3, labels = ['less than a liter', 'between one and two liter', 'more than two liter'])
obesity_df["FAF"] = pd.cut(obesity_df.FAF, bins = 4, labels = ['no physical activity', 'one or two days physical activity', 'two or four days physical activity', 'four or five days physical activity'])
obesity_df["TUE"] = pd.cut(obesity_df.TUE, bins = 3, labels = ['0-2 hours', '3-5 hours', 'more than 5 hours'])


#============ wrangling
# dummy
obesity_df = pd.get_dummies(obesity_df, drop_first=True)
obesity_df.head()


#============ variables explicatives
x = obesity_df.drop(["Gender_Male"], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = obesity_df["Gender_Male"]
y.shape

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

#============ Modele 1
xgb_c = xgb.XGBClassifier(objective='binary:logistic', missing=None, seed=42)

xgb_c.fit(x_train, y_train, verbose=True, eval_set=[(x_test, y_test)])

# predictions sur ensemble de test
y_pred = xgb_c.predict(y_test)

# score du modèle : accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# matrice de confusion
cm = confusion_matrix(y_test, y_pred) 
print(cm)

# feature importance
from matplotlib import pyplot as plt
plt.figure(figsize = (12,10))
sorted_idx = xgb_c.feature_importances_.argsort()
plt.barh(obesity_df.drop(["Gender_Male"], axis=1).columns, xgb_c.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance")


""" peut-on améliorer ce modele avec un grid search ?"""

#============ Modele 2
# definition des parametres
param_grid = {
    'n_estimators': range(6, 10),
    'max_depth': range(3, 8),
    'learning_rate': [.001, .01, .02, .2, .3, .4],
    'colsample_bytree': [.7, .8, .9, 1]
}

# instanciation du classifier
xgb_c2 = XGBClassifier()

# gread search
g_search = GridSearchCV(estimator=xgb_c2, param_grid=param_grid,
                        cv=3, n_jobs=-1, verbose=0, return_train_score=True)

# ajustement du modele
g_search.fit(x_train, y_train)

# affichage des meilleurs paramètres
print(g_search.best_params_)


# predictions sur ensemble de test
y_pred = g_search.predict(y_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# matrice de confusion
cm = confusion_matrix(y_test, y_pred) 
print(cm)


""" on n'a pas pu améliorer le modèle """






