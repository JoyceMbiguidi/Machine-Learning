#============
# ARBRES DE DECISIONS POUR LA CLASSIFICATION
# Objectif : prédire les valeurs d'une target binaire et expliquer l'impact des features sur la target
#============

#============ description des données
"""
>>> bank client data:
age (numeric)
job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
default: has credit in default? (categorical: 'no','yes','unknown')
housing: has housing loan? (categorical: 'no','yes','unknown')
loan: has personal loan? (categorical: 'no','yes','unknown')

>>> related with the last contact of the current campaign:
contact: contact communication type (categorical: 'cellular','telephone')
month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

>>> other attributes:
campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
previous: number of contacts performed before this campaign and for this client (numeric)
poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

>>> social and economic context attributes
emp.var.rate: employment variation rate - quarterly indicator (numeric)
cons.price.idx: consumer price index - monthly indicator (numeric)
cons.conf.idx: consumer confidence index - monthly indicator (numeric)
euribor3m: euribor 3 month rate - daily indicator (numeric)
nr.employed: number of employees - quarterly indicator (numeric)

>>> target
y - has the client subscribed a term deposit? (binary: 'yes','no')
"""

#============ chargement des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/bank-full.csv"
raw_df = pd.read_csv(path, sep = ";")

#============ copie du dataset brut
bank_df = raw_df
bank_df.head()

#============ vérification des types
bank_df.dtypes

#============ afficher la dimension
print(bank_df.shape)

#============ recherche des valeurs manquantes
print(bank_df.isnull().sum())

"""
Problematique : on veut expliquer et prédire les facteurs qui ont une influence sur la souscription d'un dépôt à terme
"""

#============ data wrangling
# disrétisation des variables continues
bank_df["age_disc"] = pd.cut(bank_df.age, bins = [18, 33, 48, 63])
bank_df["balance"] = pd.cut(bank_df.age, bins = [-8019, 1000, 2000, 200000])

# remplacement des valeurs catégorielles
bank_df["y"].replace(('no', 'yes'), (0, 1), inplace=True)
bank_df["job"] = bank_df["job"].replace('unknown').mode()
bank_df["education"] = bank_df["education"].replace('unknown').mode()

#============ sélection des variables
categ_feat = ["age_disc", "poutcome", "month", "contact", "loan", "housing", "balance", "default", "education", "marital", "job"]
target = ["y"]
bank_df = bank_df[target + categ_feat]

#============ dummy
bank_df = pd.get_dummies(bank_df, columns=categ_feat, drop_first=True)
bank_df.info()

#============ variables explicatives
x = bank_df.drop(["y"], axis=1)

#============ variable à expliquer
y = bank_df["y"]

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

#============ MODELE 1 - entrainement du modèle
model1 = DecisionTreeClassifier(criterion='gini')
model1.fit(x_train, y_train)

#============ export du modele sous forme de texte
from sklearn.tree import export_text
export_txt = export_text(model1, feature_names=list(x.columns))
print(export_txt)

#============ predictions sur les donnees de test
y_predict_test = model1.predict(x_test)
acc = accuracy_score(y_test, y_predict_test).round(3)
print("On obtient un Accuracy de : ", acc * 100, "% dans l'échantillon de test.")

#============ matrice de confusion
confusion_matrix(y_test, y_predict_test)


y_predict_test = model1.predict(x_test)
cm_test = confusion_matrix(y_test, y_predict_test)
cm_test_perc = confusion_matrix(y_test, y_predict_test, normalize = 'true')

acc_score_test = accuracy_score(y_test, y_predict_test)

name = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
counts = [cm_test[0][0], cm_test[0][1], cm_test[1][0], cm_test[1][1]]
percentage = [cm_test_perc[0][0]*100, cm_test_perc[0][1]*100, cm_test_perc[1][0]*100, cm_test_perc[1][1]*100]


boxlabels = [f"{n}\n{c}\n{p.round(5)}%" for n,c,p in zip(name, counts, percentage)]
boxlabels = np.asarray(boxlabels).reshape(cm_test.shape[0],cm_test.shape[1])

#plt.figure(figsize = (12,10))
plt.title('Confusion Matrix test, Accuracy = '+ str(acc_score_test.round(3)*100)+ '%')

sns.heatmap(cm_test_perc, 
            cmap = 'Blues',
            xticklabels = ['Negative', 'Positive'],
            yticklabels = ['Negative', 'Positive'],
            fmt = '',
            annot = boxlabels)
    
plt.xlabel('Predict')
plt.ylabel('Real')

plt.show()

#============ métriques du modele
print(classification_report(y_test, y_predict_test))

#============ on affiche les classes prédites de Y et ses probabilités dans l'ensemble de test
bank_df_test_ = pd.DataFrame(x_test, columns = list(x.columns))
bank_df_test_.head(10)


bank_df_test_ = bank_df_test_.assign(y_actual = y_test) # on insert les classes réelles : 0 = benin, 1 = malin

bank_df_test_ = bank_df_test_.assign(y_predicted = y_predict_test) # on insert les classes prédites : 0 = benin, 1 = malin

bank_df_test_.head(20)

probs = model1.predict_proba(x_test)[:, 1] # on affiche les probabilites de la classe d'appartenance
print(probs)

#============ importance des variables
# dictionnaire des features + importance des variables
feat_dict= {}
for col, val in sorted(zip(x_train.columns, model1.feature_importances_), key=lambda x:x[1], reverse=True):
  feat_dict[col]=val

# on convertit le dictionnaire en dataframe
feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})
feat_df

values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(15, 8))
clrs = ['lightgreen' if (x < max(values)) else 'lightblue' for x in values]
sns.barplot(y=idx,x=values,palette=clrs).set(title='Variables importantes dans le risque de churn')
plt.show()

