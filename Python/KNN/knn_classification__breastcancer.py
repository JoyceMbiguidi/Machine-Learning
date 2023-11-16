#============
# K NEAREST NEIGHBORS POUR LA CLASSIFICATION
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42, stratify = y)

#============ standardisation des données
scaler = StandardScaler()
scaler.fit(x_train)

x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)

#============ MODELE sur données standardisées
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(x_train_scale, y_train)


# Prédiction des probabilités sur l'ensemble de test
y_test_predict      = KNN_classifier.predict(x_test_scale)
y_test_probs = KNN_classifier.predict_proba(x_test_scale)[:,1] 
# les probabilités prédites sont rapportées pour les deux classes. On affiche la proba de l'achat !

np.round(KNN_classifier.predict_proba(x_test_scale),3)[:5]


#============ métriques
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

print(confusion_matrix(y_test, y_test_predict))

# rapport de classification de sklearn
print(classification_report(y_test, y_test_predict))


# métriques faites maison
def my_KNN_report(x_train_scale, y_train, x_test_scale, y_test, K=5, threshold=0.5):
    knn= KNeighborsClassifier(n_neighbors=K)
    knn.fit(x_train_scale, y_train)
    probs = knn.predict_proba(x_test_scale)[:,1]
    y_test_predict = np.where(probs>=threshold,1,0)
    
    cm = confusion_matrix(y_test, y_test_predict)
    accuracy = round(accuracy_score(y_test,y_test_predict) ,4)
    error_rate = round(1-accuracy,4)
    precision = round(precision_score(y_test,y_test_predict),2)
    recall = round(recall_score(y_test,y_test_predict),2)
    f1score = round(f1_score(y_test,y_test_predict),2)
    cm_labled = pd.DataFrame(cm, index=['Réel : negatif ','Réef : positif'], columns=['Prédit : negatif','Predit : positif'])
    
    print("-----------------------------------------")
    print('Accuracy  = {}'.format(accuracy))
    print('Error_rate  = {}'.format(error_rate))
    print('Precision = {}'.format(precision))
    print('Recall    = {}'.format(recall))
    print('f1_score  = {}'.format(f1score))
    print("-----------------------------------------")
    return cm_labled


my_KNN_report(x_train, y_train, x_test_scale, y_test, K=5, threshold=0.5)




































