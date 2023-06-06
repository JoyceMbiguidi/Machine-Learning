#============
# REGRESSION LOGISTIQUE MULTIPLE
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle binaire
#============

#============ description des données
"""
ID number
Diagnosis (M = malignant, B = benign)
radius (mean of distances from center to points on the perimeter)
texture (standard deviation of gray-scale values)
perimeter
area
smoothness (local variation in radius lengths)
compactness (perimeter^2 / area - 1.0)
concavity (severity of concave portions of the contour)
concave points (number of concave portions of the contour)
symmetry
fractal dimension ("coastline approximation" - 1)
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
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/Breast_Cancer.csv"
raw_df = pd.read_csv(path, sep = ",").drop(['id', 'Unnamed: 32'], axis = 1)

#============ copie du dataset brut
cancer_df = raw_df
cancer_df.head()

#============ vérification des types
cancer_df.dtypes

#============ afficher la dimension
print(cancer_df.shape)

#============ recherche des valeurs manquantes
print(cancer_df.isnull().sum())

#============ fréquence des modalités
cancer_df["diagnosis"].value_counts()

#============ recodage des modalités
def diagnosis(x):
    if x == 'M':
        return 1
    else:
        return 0
    
cancer_df['label'] = cancer_df['diagnosis'].apply(diagnosis)
cancer_df.head()

#============ matrice de correlation
corr_matrix = cancer_df.drop(['diagnosis', 'label'], axis = 1).corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()


"""
Problematique : on veut expliquer et prédire les facteurs qui ont une influence sur la survenue du cancer du sein
"""

#============ variables explicatives
x = cancer_df.drop(['diagnosis', 'label'], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = cancer_df['label'].to_numpy()
y.shape

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42, stratify = y)

#============ standardisation des données
scaler = StandardScaler()
scaler.fit(x_train)

x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)

#============ MODELE sur données standardisées
model = LogisticRegression()
model.fit(x_train_scale, y_train)

#============ prédictions sur les jeux d'entrainement et de test
y_predict_train = model.predict(x_train_scale)
y_predict_test = model.predict(x_test_scale)

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

#plt.figure(figsize = (12,10))
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
Vrais négatifs : négatifs correctement prédits (patient sain identifié comme sain)
Vrais positifs : positifs correctement prédits (patient malade identifié comme malade)
Faux négatifs : négatifs mal prédits (patient malade identifié comme sain)
Faux positifs : positifs prédits de manière incorrecte (patient sain identifié comme malade)


1. En haut à gauche (vrai négatif) : combien de fois le modèle a-t-il correctement classé un échantillon négatif comme négatif ?
2. En haut à droite (faux positif) : combien de fois le modèle a-t-il incorrectement classé un échantillon négatif comme positif ?
3. En bas à gauche (faux négatif) : combien de fois le modèle a-t-il classé à tort un échantillon positif comme négatif ?
4. En bas à droite (vrai positif) : combien de fois le modèle a-t-il correctement classé un échantillon positif comme positif ?
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

"""
1. Précision : la Précision est le rapport entre les Vrais Positifs et tous les Positifs proposés. 
Ici, la précision mesure les patients que nous identifions (algorithme) correctement 
comme ayant un cancer du sein parmi tous les patients qui en sont réellement atteints.
On détecte les patients malades avec beaucoup de précision si cette valeur est élevée.

2.Recall : ou rappel (sensibilité), est la mesure de notre modèle identifiant correctement les Vrais Positifs. 
Ainsi, pour tous les patients qui ont réellement un cancer du sein, le rappel nous indique combien 
nous avons correctement identifié comme ayant un cancer.
"""

"""
Nous avons besoin d'un compromis entre la précision et le rappel. 
Nous devons d'abord décider lequel est le plus important pour notre problème de classification.

En matière de santé on va privilégier le RAPPEL, car nous aimerions détecter le plus de patients possibles
atteints d'un cancer.

Il existe également de nombreuses situations où la précision et le rappel sont tout aussi importants. 
Par exemple, pour notre modèle, si le médecin nous informe que les patients qui ont été incorrectement 
classés comme atteints du cancer sont tout aussi importants car ils pourraient 
être le signe d'une autre affection, alors nous viserons non seulement un rappel élevé, 
mais un précision aussi.

Dans de tels cas, nous utilisons une métrique appelée F1-score.

Le F1-score est une métrique de classification qui mesure la capacité d'un modèle à bien prédire 
les individus positifs, tant en termes de precision (taux de prédictions positives correctes) 
qu'en termes de recall (taux de positifs correctement prédits).

C'est plus facile à travailler car maintenant, au lieu d'équilibrer précision et rappel, 
nous pouvons simplement viser un bon score F1 et cela indiquerait également une bonne précision 
et une bonne valeur de rappel.
"""

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

"""
les accuracy sont satisfaisants. Pas d'overffiting.
par curiosité on peut tenter de jouer avec les hyper paramètres pour vérifier si l'accuracy
du test set peut être amélioré.
"""

#============ importance des variables
importances = pd.DataFrame(data={
    'Attribute': cancer_df.drop(['diagnosis','label'], axis = 1).columns,
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

"""
Texture_worst contribue très fortement à la survenance du cancer du sein. Sa valeur est positive et son coefficient le plus élevé. 
Ainsi, on a 4 fois plus de chances d'avoir le Cancer du sein, lorsque la cellule est de type "Texture_worst".

On peut alors proposer au patient des aliments ou médicaments qui ralentissent la viellissement cellulaire, 
voire une pratique sportive, en plus d'un traitement adéquat si la maladie est avérée.
"""

#============
# Hyperparameter tunning
#============

from sklearn.model_selection import GridSearchCV
parameters = [{'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]
grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(x_train, y_train)
best_accuracy_log = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_accuracy_log)
print(best_parameters)