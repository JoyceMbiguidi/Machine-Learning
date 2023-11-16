#============
# SUPPORT VECTOR MACHINE POUR LA CLASSIFICATION
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle
#============

#============ description des données
"""
Nous disposons d'un ensemble de caractéristiques décrivant des bouteilles de vin rouge et blanc.
Peut-on prédire la qualité du vin en fonction de ses caractéristiques ?

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
quality : bon ou mauvais, selon une note >5 et <5
color
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
path1 = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/winequality-red.csv"
path2 = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/winequality-white.csv"

df1 = pd.read_csv(path1, sep = ";", decimal=".")
df1["color"] = "red"

df2 = pd.read_csv(path2, sep = ";", decimal=".")
df2["color"] = "white"

frames = [df1, df2]
wine_df = pd.concat(frames)
wine_df.head()

#============ vérification des types
wine_df.dtypes

#============ afficher la dimension
print(wine_df.shape)

#============ recherche des valeurs manquantes
print(wine_df.isnull().sum())

#============ description des donnees
wine_df.describe()

#============ matrice de correlation
corr_matrix = wine_df.corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()


#============ recodate de la cible
def change_values(x):
    if x > 5 :
        return 'good'
    else :
        return 'bad'

wine_df['quality'] = wine_df['quality'].apply(change_values)


"""
Problematique : Comment prédire la qualité du vin (un bon vin) en fonction de ses caractéristiques chimiques ?
"""


#============ dummy variables
categ_feat = wine_df.select_dtypes(include=['object']).columns

wine_df = pd.get_dummies(wine_df, columns=categ_feat, drop_first=True)


#============ standardisation des variables continues
scale = StandardScaler()
wine_df_scaled = scale.fit_transform(wine_df[['fixed acidity', 'volatile acidity',
                                              'citric acid', 'residual sugar', 'chlorides', 
                                              'free sulfur dioxide', 'total sulfur dioxide', 
                                              'density', 'pH', 'sulphates', 'alcohol']]) # je retire mon Y et les dummy

wine_df_scaled = pd.DataFrame(wine_df_scaled, columns = wine_df[['fixed acidity', 'volatile acidity',
                                              'citric acid', 'residual sugar', 'chlorides', 
                                              'free sulfur dioxide', 'total sulfur dioxide', 
                                              'density', 'pH', 'sulphates', 'alcohol']].columns)

wine_df_scaled = pd.concat([wine_df_scaled, wine_df[['quality_good', 'color_white']].reset_index(drop=True)], axis=1)

wine_df_scaled.head()



#============ variable explicative
x = wine_df_scaled.drop('quality_good', axis = 1)
x.shape

#============ variable à expliquer
y = wine_df_scaled['quality_good']
y.shape

sns.histplot(data = y)


#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42) #shuffle : mélange pour tirage aléatoire


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



















