#============
# EXTREME GRADIENT BOOSTING POUR LA CLASSIFICATION
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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

#============ standardisation des données
scaler = StandardScaler() # on standardise nos features (on centre et on réduit X, i.e (x-mean(x)) / sd(x) )
scaler.fit(x_train)

x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)

#============ MODELE sur données standardisées
# definition des parametres
param_grid = {
    'n_estimators': range(6, 10),
    'max_depth': range(3, 8),
    'learning_rate': [.001, .01, .02, .2, .3, .4],
    'colsample_bytree': [.7, .8, .9, 1]
}

# instanciation du classifier
xgb_c = XGBClassifier()

# gread search
g_search = GridSearchCV(estimator=xgb_c, param_grid=param_grid,
                        cv=3, n_jobs=-1, verbose=0, return_train_score=True)

# ajustement du modele
g_search.fit(x_train, y_train)

# affichage des meilleurs paramètres
print(g_search.best_params_)


# predictions sur ensemble de test
y_pred = g_search.predict(y_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

