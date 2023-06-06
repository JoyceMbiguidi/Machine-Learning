#============
# REGRESSION LOGISTIQUE MULTIPLE
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle binaire
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

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

#============ fréquence des modalités
bank_df["y"].value_counts()

#============ statistiques descriptives
bank_df.describe()
bank_df.describe(include = [object]).transpose()

#============ dataviz
ax = bank_df["age"].hist(bins=20, density=False, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='age')
plt.show()

ax = bank_df["balance"].hist(bins=80, density=False, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='balance')
plt.xlim(-3000, 10000)
plt.show()

ax = bank_df["campaign"].hist(bins=80, density=False, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='campaign')
plt.xlim(0, 30)
plt.show()

#============ data wrangling
# disrétisation des variables continues
bank_df["age_disc"] = pd.cut(bank_df.age, bins = [18, 33, 48, 63])
bank_df["balance"] = pd.cut(bank_df.age, bins = [-8019, 1000, 2000, 200000])

# remplacement des valeurs catégorielles
bank_df["y"].replace(('no', 'yes'), (0, 1), inplace=True)
bank_df["job"] = bank_df["job"].replace('unknown').mode()
bank_df["education"] = bank_df["education"].replace('unknown').mode()

"""
Problematique : on veut expliquer et prédire les facteurs qui ont une influence sur la souscription d'un dépôt à terme
"""

#============ sélection des variables
categ_feat = ["age_disc", "poutcome", "month", "contact", "loan", "housing", "balance", "default", "education", "marital", "job"]
target = ["y"]
bank_df = bank_df[target + categ_feat]

#============ dummy
bank_df = pd.get_dummies(bank_df, columns=categ_feat, drop_first=True)
bank_df.info()

#============ variables explicatives
x = bank_df.drop(["y"], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = bank_df["y"]
y.shape

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

#============ standardisation des données
"""
Pas de standardscaler sur dummy
"""

#============ MODELE sur données non standardisées
model = LogisticRegression()
model.fit(x_train, y_train)

#============ prédictions sur les jeux d'entrainement et de test
y_predict_train = model.predict(x_train)
y_predict_test = model.predict(x_test)

#============ évaluation du modèle sur le jeu d'entrainement
from sklearn.metrics import confusion_matrix

cm_train = confusion_matrix(y_train, y_predict_train)
cm_train_perc = confusion_matrix(y_train, y_predict_train, normalize = 'true')

acc_score_train = accuracy_score(y_train, y_predict_train)

name = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
counts = [cm_train[0][0], cm_train[0][1], cm_train[1][0], cm_train[1][1]]
percentage = [cm_train_perc[0][0]*100, cm_train_perc[0][1]*100, cm_train_perc[1][0]*100, cm_train_perc[1][1]*100]


boxlabels = [f"{n}\n{c}\n{p.round(2)}%" for n,c,p in zip(name, counts, percentage)]
boxlabels = np.asarray(boxlabels).reshape(cm_train.shape[0],cm_train.shape[1])

plt.title('Confusion Matrix Train, Accuracy = '+ str(acc_score_train.round(2)*100)+ '%')

sns.heatmap(cm_train_perc, 
            cmap = 'Blues',
            xticklabels = ['Negative', 'Positive'],
            yticklabels = ['Negative', 'Positive'],
            fmt = '',
            annot = boxlabels)
    
plt.xlabel('Predict')
plt.ylabel('Real')

plt.show()

"""
Vrais négatifs : négatifs correctement prédits (client non souscripteur dont le modèle a correctement prédit sa non-souscription)
Vrais positifs : positifs correctement prédits (client souscripteur dont le modèle a correctement prédit la souscription)
Faux négatifs : positifs prédits à tort comme négatifs (client souscripteur dans la réalité dont le modèle a prédit une non-souscription)
Faux positifs : négatifs prédits à tort comme positifs (client non-souscripteur que le modèle a prédit comme ayant souscrit)


1. En haut à gauche (vrai négatif) : combien de fois le modèle a-t-il correctement classé un échantillon négatif comme négatif ?
2. En haut à droite (faux positif) : combien de fois le modèle a-t-il incorrectement classé un échantillon négatif comme positif ?
3. En bas à gauche (faux négatif) : combien de fois le modèle a-t-il classé à tort un échantillon positif comme négatif ?
4. En bas à droite (vrai positif) : combien de fois le modèle a-t-il correctement classé un échantillon positif comme positif ?
"""

"""
ACCURACY
C'est la somme de tous les vrais positifs et vrais négatifs qu'il divise par le nombre total d'instances : (194 + 7885) / 9043 = 0.89. 
Il permet d'apporter une réponse à la question suivante : 
    de toutes les classes positives et négatives, combien parmi elles ont été prédites correctement ? 
    Des valeurs élevées de ce paramètre sont souvent souhaitables. 
"""

"""
PRECISION
La précision indique le rapport entre les prévisions positives correctes et le nombre total de prévisions positives :
    TP / (FP + TP)
Ce paramètre répond donc à la question suivante : 
    sur tous les enregistrements positifs prédits, combien sont réellement positifs ? 
Autrement dit, cette métrique mesure à quel point notre modèle a vu juste, à chaque fois qu'il a fait une bonne prédiction.
"""

"""
RECALL
Le rappel (ou recall) est un paramètre qui permet de mesurer le nombre de prévisions positives correctes 
sur le nombre total de données positives. Recall = TruePositives / (TruePositives + FalseNegatives)
Il permet de répondre à la question suivante : 
    sur tous les enregistrements positifs, combien ont été correctement prédits ? 
"""

"""
F1 SCORE
Le score F1 est une moyenne harmonique de la précision et du rappel. 
Sa valeur est maximale lorsque le rappel et la précision sont équivalents.

La métrique score F1 est alors utilisée pour évaluer la performance de l'algorithme. 
De même, il est particulièrement difficile de comparer deux modèles avec une faible précision et un rappel élevé. 
Dans ces conditions, le score F1 permet de mesurer ces deux paramètres simultanément.
"""

#============ métriques du jeu d'entrainement
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#precision - how many were actually positive?
precision_train = precision_score(y_train, y_predict_train)

#recall - True Positive Rate
recall_train = recall_score(y_train, y_predict_train)

#F1 score
F1_train = f1_score(y_train, y_predict_train)

print("Training Accuracy = ", acc_score_train.round(3))
print("Training Precision = ", precision_train.round(3))
print("Training Recall = ", recall_train.round(3))
print("F1 train = ", F1_train.round(3))

#============ évaluation du modèle sur le jeu de test
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

#============ métriques du jeu de test
#precision - how many were actually positive?
precision_test = precision_score(y_test, y_predict_test)

#recall - True Positive Rate
recall_test = recall_score(y_test, y_predict_test)

#F1 score
F1_test = f1_score(y_test, y_predict_test)

print("Testing Accuracy = ", acc_score_test.round(3))
print("Testing Precision = ", precision_test.round(3))
print("Testing Recall = ", recall_test.round(3))
print("F1 test = ", F1_train.round(3))

#============ table des métriques
metrics = {}
metrics['accuracy'] = (acc_score_train.round(3),
                  acc_score_test.round(3))
metrics['precision'] = (precision_train.round(3),
                  precision_test.round(3))
metrics['recall'] = (recall_train.round(3),
                  recall_test.round(3))
metrics['F1 score'] = (F1_train.round(3),
                  F1_test.round(3))
metrics_df = pd.DataFrame(metrics).transpose()
metrics_df.columns = ['Train', 'Test']
print(metrics_df)

#============ importance des variables
importances = pd.DataFrame(data={
    'Attribute': bank_df.drop(['y'], axis = 1).columns,
    'Coefficients': model.coef_[0].round(3),
    'Exponentielle': np.exp(model.coef_[0]).round(2)})

importances = importances.sort_values(by='Coefficients', ascending=False)
importances

"""
1. on regarde le signe du coefficient. Si + alors il contribue à la survenance de la maladie. Sinon, effet contraire, il contribue à limiter voire contrecarrer la maladie.
2. on regarde la valeur de l'exponentielle et si EXP > 1, alors on a plus de chances d'être POSITIF, sinon on a moins de chances d'être POSITIF à la maladie
"""

#plt.figure(figsize = (20, 9))
plt.bar(x=importances['Attribute'], height=importances['Coefficients'], color='#069AF3')
plt.title('Importance des variables classées selon leurs coefs', size=20)
plt.xticks(rotation='vertical')
plt.show()