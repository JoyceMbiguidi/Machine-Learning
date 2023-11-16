#============
# RANDOM FOREST POUR LA CLASSIFICATION
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle
#============

#============ description des données
"""
On dispose de données médicales de 53 patients. 
L'objectif est de prédire qui est atteint ou non du cancer de la prostate.

Age = âge du patient au moment du diagnostic
Acide = niveau d'acide phosphatase sérique (protéine ?)
Rayonx = résultat d'une analyse par rayonX (X=0, négatif, 1=positif)
Taille = la taille de la tumeur (0 = petite, 1=grande)
Grade = l'état de la tumeur déterminé par biopsie (0=moyen, 1=grave) (prélèvement?)
Log.acide = logarithme népérien du niveau d'acidité
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
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/cancerprostate.txt"
raw_df = pd.read_csv(path, sep = ";").drop(['log.acid'], axis = 1)

#============ copie du dataset brut
cancer_df = raw_df
cancer_df.head()

#============ vérification des types
cancer_df.dtypes

#============ afficher la dimension
print(cancer_df.shape)

#============ recherche des valeurs manquantes
print(cancer_df.isnull().sum())


"""
Problematique : on veut expliquer et prédire les facteurs qui ont une influence sur la survenue du cancer de la prostate
"""

#============ variables explicatives
x = cancer_df.drop(['Y'], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = cancer_df['Y'].to_numpy()
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
cancer_df_test_ = pd.DataFrame(x_test, columns = list(cancer_df.drop(['Y'], axis=1).columns))
cancer_df_test_.head(10)


cancer_df_test_ = cancer_df_test_.assign(y_actual = y_test) # on insert les classes réelles : 0 = sain, 1 = malade

cancer_df_test_ = cancer_df_test_.assign(y_predicted = y_predict_test) # on insert les classes prédites : 0 = sain, 1 = malade

cancer_df_test_.head(20)

probs = model1.predict_proba(x_test)[:, 1] # on affiche les probabilites de la classe d'appartenance
print(probs)

#============ rapport des metriques
print(classification_report(y_test, y_predict_test))

#============ feature importance
feature_imp = pd.Series(model1.feature_importances_,index=cancer_df.drop(['Y'], axis=1).columns).sort_values(ascending=False)
feature_imp

#============ creation du graphique
values = model1.feature_importances_
plt.figure(figsize=(10, 2))
clrs = ['lightgreen' if (x < max(values)) else 'lightblue' for x in values]
sns.barplot(x=feature_imp, y=feature_imp.index, palette=clrs)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('RF : Variables importantes dans la survenue du cancer de la prostate')
plt.show()


#============ MODELE 2 : on va tenter d'ameliorer le modele en discretisant les donnees
# changement du type de variables
cancer_df["rayonx"].replace((0, 1), ("négatif", "positif"), inplace=True) # recodage des variables
cancer_df["taille"].replace((0, 1), ("petite", "grande"), inplace=True) # recodage des variables
cancer_df["grade"].replace((0, 1), ("moyen", "grave"), inplace=True) # recodage des variables

# Disrétisation des variables age + acide
cancer_df["age_disc"] = pd.cut(cancer_df.age, bins = 5)
#cancer_df["acide_disc"] = pd.cut(cancer_df.acide, bins = 3)
cancer_df["acide_disc"] = pd.cut(cancer_df.acide, bins = 3, labels = ["faible", "moyenne", "élevée"])

#============ variables explicatives
x = cancer_df.drop(["Y", "age", "acide"], axis=1)
x = pd.get_dummies(x)
x.shape

#============ variable à expliquer
y = cancer_df['Y'].to_numpy()
y.shape

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42, stratify = y)

# nouveau modele
model2 = RandomForestClassifier()
model2.fit(x_train, y_train)

#============ prédictions sur les jeux d'entrainement et de test
y_predict_train = model2.predict(x_train)
y_predict_test = model2.predict(x_test)

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


#============ rapport des metriques
print(classification_report(y_test, y_predict_test))

#============ feature importance
feature_imp = pd.Series(model2.feature_importances_, index= x.columns).sort_values(ascending=False)
feature_imp

#============ creation du graphique
values = model2.feature_importances_
plt.figure(figsize=(10, 4))
clrs = ['lightgreen' if (x < max(values)) else 'lightblue' for x in values]
sns.barplot(x=feature_imp, y=feature_imp.index, palette=clrs)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('RF : Variables importantes dans la survenue du cancer de la prostate')
plt.show()