#============
# REGRESSION LOGISTIQUE MULTIPLE
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle binaire
#============

#============ description des données
"""
survival - Survival (0 = No; 1 = Yes)
class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
name - Name
sex - Sex
age - Age
sibsp - Number of Siblings/Spouses Aboard
parch - Number of Parents/Children Aboard
ticket - Ticket Number
fare - Passenger Fare
cabin - Cabin
embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat - Lifeboat (if survived)
body - Body number (if did not survive and body was recovered)
"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
from matplotlib import axes
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/titanic.csv"
raw_df = pd.read_csv(path, sep = ",")

#============ copie du dataset brut
titanic_df = raw_df
titanic_df.head()

#============ vérification des types
titanic_df.dtypes

#============ afficher la dimension
print(titanic_df.shape)

#============ fréquence des modalités
titanic_df["Survived"].value_counts()

#============ statistiques descriptives
titanic_df.describe()
titanic_df.describe(include = [object]).transpose()

#============ sélection des variables
numeric_feat = ["Age", "SibSp", "Parch", "Fare"]
categ_feat = ["Pclass", "Sex", "Embarked"]

titanic_df = titanic_df[numeric_feat + categ_feat + ["Survived"]]
titanic_df

#============ recherche des valeurs manquantes
print(titanic_df.isnull().sum())

#============ imputation des valeurs manquantes
titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].median())
titanic_df.info()

vax = titanic_df["Age"].hist(bins=20, density=True, stacked=True, color='teal', alpha=0.6)
titanic_df["Age"].plot(kind='density', color='teal')
axes.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("missing")
titanic_df["Embarked"].value_counts()

titanic_df["Pclass"].value_counts()

#============ dummy
titanic_df = pd.get_dummies(titanic_df, columns=categ_feat, drop_first=True)
titanic_df.head()

#============ variables explicatives
x = titanic_df.drop(["Survived"], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = titanic_df["Survived"]
y.shape

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

#============ standardisation des données
scaler = StandardScaler() # on standardise nos features (on centre et on réduit X, i.e (x-mean(x)) / sd(x) )
scaler.fit(x_train)

x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)

#============ MODELE sur données standardisées
model = LogisticRegression() # initialisation du modèle LOGIT
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
Une accuracy de 81 %, fait référence à la précision de notre modèle de classification. 
L'accuracy ou précision, évalue la performance du modèle qui vient d'effectuer des prédictions.

L'accuracy mesure la proportion des prédictions correctes faites par le modèle par rapport au nombre 
total d'échantillons de données. Dans le cas d'une accuracy de 81 %, cela signifie que le modèle a 
correctement classé 81 % des échantillons de données qu'il a traités. 
En d'autres termes, il a fait des prédictions correctes pour 81 % des cas, 
tandis que 19 % des prédictions étaient incorrectes.

Il est important de noter que l'accuracy seule ne donne pas toujours une image complète de la 
performance d'un modèle. Dans certaines situations, d'autres mesures telles que la précision, 
le rappel, la F1-score ou la matrice de confusion peuvent être nécessaires pour évaluer plus en 
détail la performance du modèle, en particulier lorsque les classes ne sont pas équilibrées ou que 
certaines erreurs sont plus coûteuses que d'autres.
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
>> PRECISION:
Un "precision_score" de 78 % fait référence à la précision de notre modèle de classification. 
La précision évalue la capacité du modèle à faire des prédictions correctes parmi les exemples 
qu'il a classés comme positifs (vrais positifs) par rapport à l'ensemble des exemples qu'il a classés 
comme positifs (vrais positifs + faux positifs).

Une précision de 78 %" signifie que le modèle a correctement classé 78 % des survivants. 
Cela indique que le modèle a une capacité relativement élevée à éviter de faire de fausses prédictions positives. 
En d'autres termes, lorsque le modèle prédit que quelqu'un a survécu, il a raison dans environ 78 % des cas.


>> RECALL:
Le rappel est une mesure de performance du modèle de classification.
Le rappel est une mesure qui évalue la capacité de notre modèle à identifier 
la totalité des exemples positifs (personnes ayant survécues) dans un ensemble de données. 
Plus précisément, le rappel mesure la proportion des réellement positifs (vrais survivants) que 
le modèle a correctement identifiés par rapport au nombre total de survivants (vrais ou faux) présents dans les données.

Un rappel de 71 % signifie que le modèle a correctement identifié 71 % des survivants dans l'ensemble de données, 
ce qui est généralement considéré comme un bon résultat. 
Cependant, cela signifie également que 29 % de survivants n'ont pas été correctement identifiés par le modèle.


>> F1_SCORE
Un F1-score mesure la performance du modèle de classification. 
Le F1-score est calculé à partir de deux autres métriques de performance : la précision (precision) et le rappel (recall).

Le F1-score est une métrique particulièrement utile lorsque les classes que vous essayez de prédire ne sont pas équilibrées, 
c'est-à-dire lorsque l'une des classes est beaucoup plus fréquente que l'autre. 
Il combine la précision et le rappel en une seule valeur qui tient compte à la fois des vrais positifs 
(prédictions correctes de la classe positive), des faux positifs (prédictions incorrectes de la classe positive) 
et des faux négatifs (cas où la classe positive réelle n'a pas été prédite correctement).

Un F1-score de 74 % indique que le modèle a une performance équilibrée en termes de précision et de rappel, 
ce qui signifie qu'il est capable de faire des prédictions correctes pour la classe des survivants (positifs) 
tout en minimisant les faux positifs (personnes  décédées prédites comme ayant survécues). 
Plus le F1-score est élevé, meilleure est la performance du modèle en termes de classification.
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

"""
Vrais négatifs : négatifs correctement prédits (voyageur décédé dont le modèle a correctement prédit le décès)
Vrais positifs : positifs correctement prédits (voyageur ayant survécu et dont le modèle a correctement prédit la survie)
Faux négatifs : positifs prédits à tort comme négatifs (voyageur ayant survécu dont le modèle a prédit le décès)
Faux positifs : négatifs prédits à tort comme positifs (voyageur décédé que le modèle a prédit comme ayant survécu)


1. En haut à gauche (vrai négatif) : combien de fois le modèle a-t-il correctement classé un échantillon négatif comme négatif ?
2. En haut à droite (faux positif) : combien de fois le modèle a-t-il incorrectement classé un échantillon négatif comme positif ?
3. En bas à gauche (faux négatif) : combien de fois le modèle a-t-il classé à tort un échantillon positif comme négatif ?
4. En bas à droite (vrai positif) : combien de fois le modèle a-t-il correctement classé un échantillon positif comme positif ?

"""

"""
ACCURACY
C'est la somme de tous les vrais positifs et vrais négatifs qu'il divise par le nombre total d'instances : (11 + 108) / 179 = 0.66. 
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
Autrement dit, c'est le rapport entre les voyageurs qui ont survécu et les voyageurs qui ont survécu + ceux qui n'ont pas survécus,
mais qui ont été identifiés comme tels.
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

"""
Comprendre la précision nous a fait comprendre que nous avons besoin d'un compromis entre 
la précision et le rappel. Nous devons d'abord décider lequel est le plus important pour 
notre problème de classification.

Par exemple, pour notre ensemble de données, nous pouvons considérer qu'il est plus important 
d'obtenir un rappel élevé que d'obtenir une précision élevée - nous aimerions détecter autant de 
passagers ayant survécu. Pour certains autres modèles, comme classer si un client de la 
banque est en défaut de paiement ou non, il est souhaitable d'avoir une grande précision car la 
banque ne voudrait pas perdre des clients qui se sont vu refuser un prêt sur la base de la 
prédiction du modèle selon laquelle ils seraient défaillants.
"""

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
print("F1 test = ", F1_test.round(3))

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
    'Attribute': titanic_df.drop(['Survived'], axis = 1).columns,
    'Coefficients': model.coef_[0].round(3),
    'Exponentielle': np.exp(model.coef_[0]).round(2)})

importances = importances.sort_values(by='Coefficients', ascending=False)
importances

"""
1. on regarde le signe du coefficient. Si + alors il contribue aux facteurs de survie. 
Sinon, effet contraire, il contribue à limiter voire contrecarrer les chances de survie.
2. on regarde la valeur de l'exponentielle et si EXP > 1, alors on a plus de chances d'être POSITIF, 
sinon on a moins de chances d'être POSITIF, i.e de survivre.
"""

#plt.figure(figsize = (20, 9))
plt.bar(x=importances['Attribute'], height=importances['Coefficients'], color='#069AF3')
plt.title('Importance des variables classées selon leurs coefs', size=20)
plt.xticks(rotation='vertical')
plt.show()

