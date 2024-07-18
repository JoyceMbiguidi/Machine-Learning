#============
# SUPPORT VECTOR MACHINE POUR LA CLASSIFICATION
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle binaire
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score



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

#============ fréquence des modalités
cancer_df["Y"].value_counts()

#============ matrice de correlation
corr_matrix = cancer_df.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

# pair plot
sns.set_style('whitegrid')
sns.pairplot(cancer_df, hue='Y')


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
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                                    shuffle = True, random_state = 42, stratify = y)

#============ standardisation des données
scaler = StandardScaler()
scaler.fit(x_train)

x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)


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

1. Accuracy (Exactitude) :
   - Quand les classes sont équilibrées : Si les classes de votre ensemble de données sont à peu près équilibrées 
   (c'est-à-dire, il y a à peu près le même nombre d'exemples positifs et négatifs), alors l'accuracy peut être une 
   mesure appropriée. Elle évalue la proportion de prédictions correctes par rapport à l'ensemble des prédictions.
   
   - Quand toutes les erreurs sont équivalentes : Si vous ne pouvez pas vous permettre de privilégier les 
   faux positifs ou les faux négatifs, et que toutes les erreurs sont d'importance égale, l'accuracy peut être une bonne métrique.

2. Précision (Precision) :
   - Quand les faux positifs sont coûteux : La précision mesure la proportion de prédictions positives correctes 
   parmi toutes les prédictions positives. Si les faux positifs sont coûteux ou indésirables 
   (par exemple, dans le diagnostic médical), vous devriez privilégier la précision.

3. Rappel (Recall) :
   - Quand les faux négatifs sont coûteux : Le rappel mesure la proportion de vrais positifs parmi toutes les 
   valeurs réelles positives. Si les faux négatifs sont coûteux ou ont des conséquences graves (
       comme dans la détection de fraudes), alors le rappel est crucial.

4. F1-Score (Score F1) :
   - Quand vous avez besoin d'un équilibre entre la précision et le rappel : Le F1-Score est une métrique 
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

my_param_grid = {'C': [10,100,1000], 'gamma': ['scale',0.01,0.001], 'kernel': ['rbf', 'linear', 'poly']} 

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
# why my_SVM_report is not defined ?
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

































































