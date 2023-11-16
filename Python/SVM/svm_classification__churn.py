#============
# SUPPORT VECTOR MACHINE POUR LA CLASSIFICATION
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


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


#============ matrice de correlation
corr_matrix = churn_df.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()



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
# recodate de la cible
def change_values(x):
    if x == 'True.':
        return 1
    else :
        return 0

churn_df['Churn?'] = churn_df['Churn?'].apply(change_values)

y = churn_df["Churn?"]
y.shape

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42, stratify = y)


#============ Modele 1
from sklearn.svm import SVC
# ajustement du modele de classification sur le jeu d'entrainement
SVM_classification = SVC()
SVM_classification.fit(x_train, y_train)


# prédiction sur le jeu de test
y_predict = SVM_classification.predict(x_test)
predictions = pd.DataFrame({ 'y_test':y_test,'y_predict':y_predict})
predictions.head(15)


# évaluation du modèle sur l'ensemble de données de test
def my_SVM_report(x_train, y_train, X_test,y_test, C=1,gamma='scale' ,kernel='rbf'):
    svc= SVC(C=C, gamma=gamma, kernel=kernel)
    svc.fit(x_train, y_train)
    y_predict = svc.predict(x_test)
    
    cm = confusion_matrix(y_test, y_predict)
    accuracy = round(accuracy_score(y_test,y_predict) ,4)
    error_rate = round(1-accuracy,4)
    precision = round(precision_score(y_test,y_predict),2)
    recall = round(recall_score(y_test,y_predict),2)
    f1score = round(f1_score(y_test,y_predict),2)
    cm_labled = pd.DataFrame(cm, index=['Réel : négatif ','Réel : positif'], columns=['Prédit : negatif','Prédit : positif'])
    
    print("-----------------------------------------")
    print('Accuracy  = {}'.format(accuracy))
    print('Error_rate  = {}'.format(error_rate))
    print('Precision = {}'.format(precision))
    print('Recall    = {}'.format(recall))
    print('f1_score  = {}'.format(f1score))
    print("-----------------------------------------")
    return cm_labled


my_SVM_report(x_train, y_train, x_test, y_test, kernel='rbf')


"""
QUAND UTILISER LES METRIQUES SUIVANTES : ACCURACY, PRECISION, RAPPEL ET F1 SCORE ?

Le choix entre les mesures de performance telles que l'accuracy, la précision, le rappel et le F1-score dépend 
du contexte de votre tâche d'apprentissage automatique et des conséquences de différentes erreurs.

1. **Accuracy (Exactitude)** :
   - **Quand les classes sont équilibrées :** Si les classes de votre ensemble de données sont à peu près équilibrées 
   (c'est-à-dire, il y a à peu près le même nombre d'exemples positifs et négatifs), alors l'accuracy peut être une 
   mesure appropriée. Elle évalue la proportion de prédictions correctes par rapport à l'ensemble des prédictions.
   
   - **Quand toutes les erreurs sont équivalentes :** Si vous ne pouvez pas vous permettre de privilégier les 
   faux positifs ou les faux négatifs, et que toutes les erreurs sont d'importance égale, l'accuracy peut être une bonne métrique.

2. **Précision (Precision)** :
   - **Quand les faux positifs sont coûteux :** La précision mesure la proportion de prédictions positives correctes 
   parmi toutes les prédictions positives. Si les faux positifs sont coûteux ou indésirables 
   (par exemple, dans le diagnostic médical), vous devriez privilégier la précision.

3. **Rappel (Recall)** :
   - **Quand les faux négatifs sont coûteux :** Le rappel mesure la proportion de vrais positifs parmi toutes les 
   valeurs réelles positives. Si les faux négatifs sont coûteux ou ont des conséquences graves (
       comme dans la détection de fraudes), alors le rappel est crucial.

4. **F1-Score (Score F1)** :
   - **Quand vous avez besoin d'un équilibre entre la précision et le rappel :** Le F1-Score est une métrique 
   harmonique qui combine la précision et le rappel. Il est utile lorsque vous avez besoin d'un équilibre entre ces 
   deux métriques. C'est particulièrement important lorsque les classes sont déséquilibrées, et vous voulez éviter 
   des prédictions excessivement biaisées.

"""



"""
peut-on améliorer ce modèle ?
"""


#============ Tuning des hyper paramètres avec Gridsearch
#============ Modele 2

"""
Rappel sur quelques paramètres :
    - C représente le coût d’une mauvaise classification. Un grand C signifie que vous pénalisez les erreurs de 
    manière plus stricte, donc la marge sera plus étroite, c'est-à-dire un surajustement (petit biais, grande variance).
    
    - gamma est le paramètre libre dans la fonction de base radiale (rbf). Intuitivement, le paramètre gamma (inverse de la variance) 
    définit jusqu'où va l'influence d'un seul exemple d'entraînement, des valeurs faibles signifient « loin » et des valeurs élevées signifient « proche ».
"""

my_param_grid = {'C': [10,100,1000], 'gamma': ['scale',0.01,0.001], 'kernel': ['rbf']} 

from sklearn.model_selection import GridSearchCV

GridSearchCV(estimator=SVC(),param_grid= my_param_grid, refit = True, verbose=2, cv=5 )
model2_grid = GridSearchCV(estimator=SVC(),param_grid= my_param_grid, refit = True, verbose=2, cv=5, n_jobs=-1)


# ajustement du modele avec les hyper parametres
model2_grid.fit(x_train, y_train)

# meilleurs paramètres gridsearch
model2_grid.best_params_
model2_grid.best_estimator_

model2_y_predict_optimized = model2_grid.predict(x_test)

predictions['model2_y_predict_optimized'] = model2_y_predict_optimized
predictions.head(15)


#============ rapport des métriques
my_SVM_report(x_train, y_train, x_test, y_test, C=1000, gamma=0.001)


"""
SVM n'est pas adapté pour ce type de données
"""




"""
Métriques de SVM pour la classification :

    - Accuracy (Précision) : C'est la mesure la plus courante pour évaluer la performance d'un modèle de classification. 
    Elle indique la proportion de prédictions correctes parmi toutes les prédictions.

    - Matrice de confusion : La matrice de confusion permet de visualiser les vrais positifs, les vrais négatifs, 
    les faux positifs et les faux négatifs. Elle est utile pour évaluer la capacité de votre modèle à discriminer entre les classes.

    - Précision : La précision mesure la proportion de vrais positifs parmi les prédictions positives. 
    Elle est particulièrement utile lorsque vous voulez minimiser les faux positifs.

    - Rappel (Sensibilité) : Le rappel mesure la proportion de vrais positifs parmi tous les exemples positifs réels. 
    Il est important lorsque vous voulez minimiser les faux négatifs.

    - F1-score : L'F1-score est une métrique qui tient compte à la fois de la précision et du rappel. Elle est utile 
    lorsque vous avez besoin d'équilibrer la précision et le rappel.

    - Courbe ROC et Aire sous la courbe (AUC) : Ces métriques sont utiles pour évaluer la capacité de discrimination 
    du modèle. L'AUC mesure la capacité du modèle à classer correctement les exemples positifs et négatifs.
"""












































































