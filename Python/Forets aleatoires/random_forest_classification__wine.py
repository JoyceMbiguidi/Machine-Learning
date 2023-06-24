#============
# RANDOM FOREST POUR LA CLASSIFICATION
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle
#============

#============ description des données
"""
Nous disposons d'un ensemble de caractéristiques décrivant des bouteilles de vin rouge et blanc.
Peut-on prédire la couleur du vin en fonction de ces caractéristiques ?

fixed acidity : "Acidité fixe" se réfère à la quantité d'acides non volatils présents dans le vin, 
                notamment les acides tartrique, malique et citrique. Elle est l'un des caractères chimiques utilisés 
                pour décrire la composition du vin et contribue à son goût et à sa qualité globale.
volatile acidity : "Acidité volatile" se réfère à la quantité d'acides non volatils présents dans le vin, 
                notamment les acides tartrique, malique et citrique. Elle est l'un des caractères chimiques utilisés 
                pour décrire la composition du vin et contribue à son goût et à sa qualité globale.
citric acid : "L'acide citrique" est un acide naturellement présent dans les agrumes et est utilisé comme 
                additif alimentaire pour son goût acide. Il apporte une acidité vive et une saveur acidulée aux aliments et aux boissons.
residual sugar : quantité de sucre non fermenté qui reste dans le vin après la fermentation, 
                influençant ainsi son goût et sa perception sucrée.
sugar : Le sucre dans le vin se réfère à la quantité de sucre résiduel présente dans le vin, ce qui peut influencer 
            sa douceur, son équilibre et son profil de goût.
chlorides : présence de composés chlorés dans le vin, qui peuvent influencer son goût, son arôme et sa qualité.
free sulfur dioxide : Le dioxyde de soufre libre désigne la quantité de SO2 non combiné présente dans le vin, 
            agissant comme un agent conservateur et pouvant affecter son arôme, sa stabilité et sa qualité.
total sulfur dioxide
density : La densité dans le contexte du vin fait référence à la mesure de la masse du vin par unité de volume, 
            fournissant des indications sur sa concentration, sa viscosité et sa texture en bouche. 
            Une densité plus élevée peut indiquer un vin plus riche et plus corsé.
pH : Le pH dans le contexte du vin désigne le degré d'acidité ou d'alcalinité, influençant son équilibre gustatif 
    et sa stabilité. Un pH plus bas indique une acidité plus élevée, tandis qu'un pH plus élevé indique une plus faible acidité.
sulphates : Les sulfates dans le vin font référence aux composés chimiques contenant du soufre, tels que le dioxyde 
    de soufre, utilisés comme additifs pour diverses raisons, notamment la protection contre l'oxydation et 
    la préservation de la qualité du vin. Les sulfates peuvent également influencer l'arôme, la saveur et la stabilité du vin.
alcohol : Un taux d'alcool plus élevé peut contribuer à une sensation de chaleur en bouche et à une plus grande concentration de saveurs.
quality
color
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
from sklearn.preprocessing import StandardScaler

#============ importation des données
path1 = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/winequality-red.csv"
path2 = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/winequality-white.csv"

df1 = pd.read_csv(path1, sep = ";", decimal=".")
df1["color"] = "red"

df2 = pd.read_csv(path2, sep = ";", decimal=".")
df2["color"] = "white"

frames = [df1, df2]
df = pd.concat(frames)
df.head()

#============ vérification des types
df.dtypes

#============ afficher la dimension
print(df.shape)

#============ recherche des valeurs manquantes
print(df.isnull().sum())

#============ description des donnees
df.describe()

#============ matrice de correlation
corr_matrix = df.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

#============ recodate de la cible
def change_values(x):
    if x == "red" :
        return '1'
    else :
        return '0'

df['color_'] = df['color'].apply(change_values)
df['color_'] = pd.to_numeric(df['color_'])

pass

"""
Problematique : Peut-on prédire la couleur du vin en fonction de ces caractéristiques ?
"""

#============ variables explicatives
x = df.drop(['color','color_'], axis = 1).to_numpy()
x.shape

#============ variable à expliquer
y = df['color_'].to_numpy() 
y.shape

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42, stratify = y)


#============ standardisation des variables
scaler = StandardScaler()

scaler.fit(x_train)

x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)

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

#============ rapport des metriques
print(classification_report(y_test, y_predict_test))

#============ feature importance
feature_imp = pd.Series(model1.feature_importances_,index=df.drop(['color','color_'], axis = 1).columns).sort_values(ascending=False)
feature_imp

#============ creation du graphique
values = model1.feature_importances_
plt.figure(figsize=(10, 4))
clrs = ['lightgreen' if (x < max(values)) else 'lightblue' for x in values]
sns.barplot(x=feature_imp, y=feature_imp.index, palette=clrs)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('RF : Variables importantes dans la prédiction de la couleur de vin rouge')
plt.show()


""" peut-on rendre ce modele plus interprétable ?"""

""" tentons un modeles avec variables discrétisées. Mais avant, analysons les distributions des variables"""

#============ analyses graphique
sns.histplot(data=df, x='chlorides', hue='color')
sns.histplot(data=df, x='total sulfur dioxide', hue='color')
sns.histplot(data=df, x='volatile acidity', hue='color')
sns.histplot(data=df, x='density', hue='color')
sns.histplot(data=df, x='sulphates', hue='color')
sns.histplot(data=df, x='free sulfur dioxide', hue='color')
sns.histplot(data=df, x='fixed acidity', hue='color')
sns.histplot(data=df, x='residual sugar', hue='color')
sns.histplot(data=df, x='pH', hue='color')
sns.histplot(data=df, x='citric acid', hue='color')
sns.histplot(data=df, x='alcohol', hue='color')
sns.histplot(data=df, x='quality', hue='color')


#============ discretisons arbitrairement les variables
df["chlorides"] = pd.cut(df.chlorides, bins = 3, labels = ["faible", "moyenne", "élevée"])
df["total sulfur dioxide"] = pd.cut(df["total sulfur dioxide"], bins=3, labels=["faible", "moyenne", "élevée"])
df["volatile acidity"] = pd.cut(df["volatile acidity"], bins = 3, labels = ["faible", "moyenne", "élevée"])
df["density"] = pd.cut(df.density, bins = 3, labels = ["faible", "moyenne", "élevée"])
df["sulphates"] = pd.cut(df.sulphates, bins = 3, labels = ["faible", "moyenne", "élevée"])
df["free sulfur dioxide"] = pd.cut(df["free sulfur dioxide"], bins = 3, labels = ["faible", "moyenne", "élevée"])
df["fixed acidity"] = pd.cut(df["fixed acidity"], bins = 3, labels = ["faible", "moyenne", "élevée"])
df["residual sugar"] = pd.cut(df["residual sugar"], bins = 3, labels = ["faible", "moyenne", "élevée"])
df["pH"] = pd.cut(df.pH, bins = 3, labels = ["faible", "moyenne", "élevée"])
df["citric acid"] = pd.cut(df["citric acid"], bins = 3, labels = ["faible", "moyenne", "élevée"])
df["alcohol"] = pd.cut(df.alcohol, bins = 3, labels = ["faible", "moyenne", "élevée"])
df["quality"] = pd.cut(df.quality, bins = 2, labels = ["faible", "élevée"])


#============ dummy
df_dummies = pd.get_dummies(df.drop(["color", "color_"], axis = 1))

# concatenation
df = pd.concat([df["color_"], df_dummies], axis=1)

#============ variables explicatives
x = df.drop(['color_'], axis = 1).to_numpy()
x.shape

#============ variable à expliquer
y = df['color_'].to_numpy() 
y.shape

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42, stratify = y)


#============ MODELE 2
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
feature_imp = pd.Series(model2.feature_importances_,index=df.drop(['color_'], axis = 1).columns).sort_values(ascending=False)
feature_imp

#============ creation du graphique
values = model2.feature_importances_
plt.figure(figsize=(12, 6))
clrs = ['lightgreen' if (x < max(values)) else 'lightblue' for x in values]
sns.barplot(x=feature_imp, y=feature_imp.index, palette=clrs)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('RF : Variables importantes dans la prédiction de la couleur de vin rouge')
plt.show()


