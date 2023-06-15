#============
# ARBRES DE DECISIONS POUR LA CLASSIFICATION
# Objectif : prédire les valeurs d'une target binaire et expliquer l'impact des features sur la target
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
cancer_df.isnull().sum()

#============ recodage des variables
cancer_df["diagnosis"].replace(('M', 'B'), ("Malin", "Benin"), inplace=True)
cancer_df.head()

"""
Problematique : on veut expliquer et prédire les facteurs qui ont une influence sur la survenue du cancer du sein
"""

#============ variables explicatives
x = cancer_df.drop(['diagnosis'], axis = 1)

#============ variable à expliquer
y = cancer_df['diagnosis']

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)

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
cancer_df_test_ = pd.DataFrame(x_test, columns = list(x.columns))
cancer_df_test_.head(10)


cancer_df_test_ = cancer_df_test_.assign(y_actual = y_test) # on insert les classes réelles : 0 = benin, 1 = malin

cancer_df_test_ = cancer_df_test_.assign(y_predicted = y_predict_test) # on insert les classes prédites : 0 = benin, 1 = malin

cancer_df_test_.head(20)

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
sns.barplot(y=idx,x=values,palette=clrs).set(title='Variables importantes dans la survenue du cancer du sein')
plt.show()


