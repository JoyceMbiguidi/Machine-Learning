#============
# RANDOM FOREST POUR LA CLASSIFICATION
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle
#============

#============ description des données
"""
Le but de cet exercice est de prédire l'attrition client (churn en anglais) pour un opérateur télécom. Vous utiliserez le jeu de données telecom.csv.

La variable cible est Churn?.

Retenez les variables explicatives suivantes uniquements, parmi lesquelles vous identifierez les variables catégorielles et les variables numériques:

Day Mins : (total_day_minutes, Total minutes of day calls).
Day Calls : (total_day_calls, Total minutes of day calls).
Day Charge : (total_day_charge, Total charge of day calls).
Eve Mins : (total_eve_minutes, Total minutes of evening calls).
Eve Calls : (total_eve_calls, Total number of evening calls).
Eve Charge : (total_eve_charge, Total charge of evening calls).
Area Code : (area_code, string="area_code_AAA" where AAA = 3 digit area code).
Night Mins : (total_night_minutes, Total minutes of night calls).
Night Calls : (total_night_calls, Total number of night calls).
Night Charge : (total_night_charge, Total charge of night calls).
Intl Mins : (total_intl_minutes, Total minutes of international calls).
Intl Calls : (total_intl_calls, Total number of international calls).
Intl Charge : (total_intl_charge, Total charge of international calls)
Int'l Plan : (international_plan, (yes/no). The customer has international plan).
VMail Plan : (voice_mail_plan, (yes/no). The customer has voice mail plan).
CustServ Calls : (number_customer_service_calls, Number of calls to customer service)
"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/telecom.csv"
raw_df = pd.read_csv(path, sep = ",")

#============ copie du dataset brut
churn_df = raw_df
churn_df.head()

#============ vérification des types
churn_df.dtypes

#============ afficher la dimension
print(churn_df.shape)

#============ recherche des valeurs manquantes
print(churn_df.isnull().sum())

#============ description des donnees
churn_df.describe()

#============ regroupement des variables par type
numeric_feat = [
    "Day Mins",
    "Day Calls",
    "Day Charge",
    "Eve Mins",
    "Eve Calls",
    "Eve Charge",
    "Night Mins",
    "Night Calls",
    "Night Charge",
    "Intl Mins",
    "Intl Calls",
    "Intl Charge",
    "CustServ Calls"
] # on définit la liste des variables numériques

categ_feat = [
    "Area Code",
    "Int'l Plan",
    "VMail Plan"
] # on définit la liste des variables qualitatives


"""
Problematique : on veut expliquer et prédire les facteurs qui ont une influence sur le churn
"""

#============ variables explicatives
x = churn_df[numeric_feat + categ_feat]
x = pd.get_dummies(x, columns=categ_feat, drop_first=True)
x.shape

#============ variable à expliquer
y = churn_df["Churn?"]
y.shape

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42, stratify = y)

#============ MODELE 1
model1 = RandomForestClassifier()
model1.fit(x_train, y_train)

#============ prédictions sur les jeux d'entrainement et de test
y_predict_train = model1.predict(x_train)
y_predict_test = model1.predict(x_test)

#============ évaluation du modèle sur le jeu d'entrainement
cm_test = confusion_matrix(y_test, y_predict_test)
cm_test_perc = confusion_matrix(y_test, y_predict_test, normalize = 'true')
acc_score_test = accuracy_score(y_test, y_predict_test)

name = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
counts = [cm_test[0][0], cm_test[0][1], cm_test[1][0], cm_test[1][1]]
percentage = [cm_test_perc[0][0]*100, cm_test_perc[0][1]*100, cm_test_perc[1][0]*100, cm_test_perc[1][1]*100]


boxlabels = [f"{n}\n{c}\n{p.round(2)}%" for n,c,p in zip(name, counts, percentage)]
boxlabels = np.asarray(boxlabels).reshape(2,2)

#plt.figure(figsize = (12,10))
plt.title('Confusion Matrix Test, Accuracy = '+ str(acc_score_test.round(2)*100)+ '%')

sns.heatmap(cm_test_perc, 
            cmap = 'Blues',
            xticklabels = ['Negative', 'Positive'],
            yticklabels = ['Negative', 'Positive'],
            fmt = '',
            annot = boxlabels)
    
plt.xlabel('Predict')
plt.ylabel('Real')

plt.show()

#============ on affiche les classes prédites de Y et ses probabilités dans l'ensemble de test
churn_df_test_ = pd.DataFrame(x_test, columns = list(x.columns))
churn_df_test_.head(10)


churn_df_test_ = churn_df_test_.assign(y_actual = y_test)

churn_df_test_ = churn_df_test_.assign(y_predicted = y_predict_test)

churn_df_test_ = churn_df_test_.assign(probs = model1.predict_proba(x_test)[:, 1])

churn_df_test_.head(20)

#============ rapport des metriques
print(classification_report(y_test, y_predict_test))

#============ feature importance
feature_imp = pd.Series(model1.feature_importances_,index=x.columns).sort_values(ascending=False)
feature_imp

#============ creation du graphique
values = model1.feature_importances_
plt.figure(figsize=(10, 4))
clrs = ['lightgreen' if (x < max(values)) else 'lightblue' for x in values]
sns.barplot(x=feature_imp, y=feature_imp.index, palette=clrs)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('RF : Variables importantes dans le risque de churn')
plt.show()

""" peut-on améliorer le modele ? """


#============ tunning des hyperparametres
from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap': [True],
    'max_depth': [None, 80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5, 10],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}


#============ initialisation du modele
rfc = RandomForestClassifier()

#============ instanciation du model avec grid search
grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

#============ fitting
grid_search.fit(x_train, y_train)

#============ affichage des meilleurs parametres et l'accuracy correspondant
best_params = grid_search.best_params_
best_score = grid_search.best_score_

#============ evaluation du meilleur modele sur le jeu de test
best_model = grid_search.best_estimator_
test_score = best_model.score(x_test, y_test)

print("Best parameters:", best_params)
print("Best score:", best_score)
print("Test score:", test_score)
