#============
# ARBRES DE DECISIONS POUR LA CLASSIFICATION
# Objectif : prédire les valeurs d'une target binaire et expliquer l'impact des features sur la target
#============

#============ description des données
"""
On dispose de données médicales de 53 patients. 
L'objectif est de prédire si le cancer a atteint le réseau lymphatique.

Le réseau lymphatique est un système de vaisseaux et de ganglions qui travaillent en 
collaboration avec le système circulatoire pour transporter la lymphe, un liquide clair 
contenant des cellules immunitaires, les déchets cellulaires et les substances étrangères, à travers le corps. 
Il joue un rôle essentiel dans la réponse immunitaire, l'élimination des toxines et le maintien de 
l'équilibre des fluides corporels.

Y = [1] le cancer a atteint le réseau lymphatique, [0] le cancer n'a pas atteint le réseau lymphatique.
Age = âge du patient au moment du diagnostic
Acide = niveau d'acide phosphatase sérique
Rayonx = résultat d'une analyse par rayonX (X=0, négatif, 1=positif)
Taille = la taille de la tumeur (0 = petite, 1=grande)
Grade = l'état de la tumeur déterminé par biopsie (0=moyen, 1=grave)
Log.acide = logarithme népérien du niveau d'acidité

Lorsque le cancer atteint le réseau lymphatique, il peut se propager à d'autres parties du corps 
par le biais de la circulation lymphatique. Les cellules cancéreuses peuvent envahir les vaisseaux lymphatiques 
et se déplacer vers les ganglions lymphatiques adjacents ou vers des ganglions plus éloignés. 
Ce processus est appelé métastase lymphatique. La présence de cellules cancéreuses dans les ganglions lymphatiques 
peut indiquer un stade avancé de la maladie et nécessite une attention médicale approfondie. Les ganglions lymphatiques 
touchés peuvent être enlevés chirurgicalement ou traités par d'autres méthodes, telles que la radiothérapie 
ou la chimiothérapie, en fonction du type et de l'étendue du cancer.
"""

#============ vérifier la propreté du code
# pip install flake8P
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/cancerprostate.txt"
raw_df = pd.read_csv(path, sep = ";")

#============ copie du dataset brut
prostate_df = raw_df
prostate_df.head()

#============ vérification des types
prostate_df.dtypes

#============ afficher la dimension
print(prostate_df.shape)

#============ recherche des valeurs manquantes
prostate_df.isnull().sum()

"""
Problematique : quels facteurs mettent en évidence la présence du cancer ?
"""

#============ recodage des variables
prostate_df["Y"].replace((0, 1), ("non atteint", "atteint"), inplace=True)
prostate_df.head()

#============ variables explicatives
x = prostate_df.drop(['Y', 'log.acid'], axis = 1)

#============ variable à expliquer
y = prostate_df['Y']

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)

#============ MODELE 1 - entrainement du modèle
model1 = DecisionTreeClassifier(criterion='gini')
model1.fit(x_train, y_train)

#============ visuel de l'arbre
plt.figure(figsize=(35,25))
plot_tree(model1, feature_names=list(x.columns), filled=True, class_names=model1.classes_)
pass

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
prostate_df_test_ = pd.DataFrame(x_test, columns = ['age', 'acide', 'rayonx', 'taille', 'grade'])
prostate_df_test_.head(10)


prostate_df_test_ = prostate_df_test_.assign(y_actual = y_test) # on insert les classes réelles : 0 = non atteint, 1 = atteint

prostate_df_test_ = prostate_df_test_.assign(y_predicted = y_predict_test) # on insert les classes prédites : 0 = non atteint, 1 = atteint

prostate_df_test_.head(20)

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

"""
peut-être sera t-il préférable de découper les variables en classes ?
"""

values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(10, 2))
clrs = ['lightgreen' if (x < max(values)) else 'lightblue' for x in values]
sns.barplot(y=idx,x=values,palette=clrs).set(title='Variables importantes dans la survenue du cancer de la prostate')
plt.show()


#============ MODELE 2 - entrainement du modèle
# changement du type de variables
prostate_df["rayonx"].replace((0, 1), ("négatif", "positif"), inplace=True)
prostate_df["taille"].replace((0, 1), ("petite", "grande"), inplace=True)
prostate_df["grade"].replace((0, 1), ("moyen", "grave"), inplace=True)

# disrétisation des variables age + acide
prostate_df["age_disc"] = pd.cut(prostate_df.age, bins = 5)
prostate_df["acide_disc"] = pd.cut(prostate_df.acide, bins = 3)
prostate_df["acide_disc"] = pd.cut(prostate_df.acide, bins = 3, labels = ["faible", "moyenne", "élevée"])

#============ variables explicatives
x = prostate_df.drop(['Y', 'log.acid', 'age', 'acide'], axis = 1)
x = pd.get_dummies(x)

#============ variable à expliquer
y = prostate_df['Y']

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
x_train.info()

model2 = DecisionTreeClassifier(criterion='gini')
model2.fit(x_train, y_train)

#============ visuel de l'arbre du modele 2
plt.figure(figsize=(35,25))
plot_tree(model2, feature_names=list(x.columns), filled=True, class_names=str(model2.classes_))
pass

#============ predictions sur les donnees de test
y_predict_test = model2.predict(x_test)
acc = accuracy_score(y_test, y_predict_test).round(3)
print("On obtient un Accuracy de : ", acc * 100, "% dans l'échantillon de test.")

#============ matrice de confusion
confusion_matrix(y_test, y_predict_test)


y_predict_test = model2.predict(x_test)
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

#============ metriques du modele 2
print(classification_report(y_test, y_predict_test))

#============ importance des variables
# dictionnaire des features + importance des variables
feat_dict= {}
for col, val in sorted(zip(x_train.columns, model2.feature_importances_), key=lambda x:x[1], reverse=True):
  feat_dict[col]=val

# on convertit le dictionnaire en dataframe
feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})
feat_df

values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(20, 8))
clrs = ['lightgreen' if (x < max(values)) else 'lightblue' for x in values]
sns.barplot(y=idx,x=values,palette=clrs).set(title='Variables importantes dans la survenue du cancer de la prostate')
plt.show()

"""
Le modele 2 a moins bien performé sur le plan des métriques, mais il a l'avantage d'être interprétable.
On comprend mieux les feature après les avoir discrétiser. Rien ne sert de courir après les métriques.
Donner du sens à l'analyse est bien mieux !
"""

