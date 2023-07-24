#============
# EXTREME GRADIENT BOOSTING POUR LA CLASSIFICATION
# Objectif : expliquer et prédire les valeurs d'une variable catégorielle binaire
#============

#============ description des données
"""
CustomerID: A unique ID that identifies each customer.
Gender: The customer’s gender: Male, Female
Age: The customer’s current age, in years, at the time the fiscal quarter ended.
Senior Citizen: Indicates if the customer is 65 or older: Yes, No
Partner: Indicates if the customer is married: Yes, No
Dependents: Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.
Number of Dependents: Indicates the number of dependents that live with the customer.
Tenure (in Months): Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
Phone Service: Indicates if the customer subscribes to home phone service with the company: Yes, No
Multiple Lines: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No
Internet Service: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
Online Security: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
Online Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
Device Protection Plan: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
Tech Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
Streaming TV: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.
Streaming Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.
Contract: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
Paperless Billing: Indicates if the customer has chosen paperless billing: Yes, No
Payment Method: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check
Monthly Charge: Indicates the customer’s current total monthly charge for all their services from the company.
Total Charges: Indicates the customer’s total charges, calculated to the end of the quarter specified above.
"""

#============ vérifier la propreté du code
# pip install flake8
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/telco_customer_churn.csv"
raw_df = pd.read_csv(path, sep = ";")

#============ copie du dataset brut
telco_df = raw_df
telco_df.head()

#============ vérification des types
telco_df.dtypes

#============ afficher la dimension
print(telco_df.shape)

#============ recherche des valeurs manquantes
print(telco_df.isnull().sum())

#============ wrangling
telco_df.drop(["customerID"], axis=1, inplace=True)
telco_df.head()

telco_df.loc[telco_df["Churn"] == "Yes", "Churn"] = 1
telco_df.loc[telco_df["Churn"] == "No", "Churn"] = 0
telco_df.head()

# on verifie le type des colonnes
telco_df.dtypes

# conversion de type
telco_df['TotalCharges'] = pd.to_numeric(telco_df['TotalCharges'])

# on remplace les espaces par des underscores
telco_df.replace(' ', '_', regex=True, inplace=True)
telco_df.head()

#============ sélection des variables
numeric_feat = telco_df.select_dtypes(include=np.number)
categ_feat = telco_df.select_dtypes(include=object)

#============ dummy
telco_df = pd.get_dummies(telco_df, columns=categ_feat.columns, drop_first=True)
telco_df.head()

#============ variables explicatives
x = telco_df.drop(["Churn"], axis=1).to_numpy()
x.shape

#============ variable à expliquer
y = telco_df["Churn"]
y.shape

#============ séparation des données : train - test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

#============ standardisation des données
scaler = StandardScaler() # on standardise nos features (on centre et on réduit X, i.e (x-mean(x)) / sd(x) )
scaler.fit(x_train)

x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)

#============ Modele 1
xgb_c = xgb.XGBClassifier(objective='binary:logistic', missing=None, seed=42)

xgb_c.fit(x_train, y_train, verbose=True, eval_set=[(x_test, y_test)])

# predictions sur ensemble de test
y_pred = xgb_c.predict(y_test)

# score du modèle : accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# matrice de confusion
cm = confusion_matrix(y_test, y_pred) 
print(cm)

# feature importance
from matplotlib import pyplot as plt
plt.figure(figsize = (12,10))
sorted_idx = xgb_c.feature_importances_.argsort()
plt.barh(telco_df.drop(["Churn"], axis=1).columns, xgb_c.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance")


""" peut-on améliorer ce modele avec un grid search ?"""

#============ Modele 2
# definition des parametres
param_grid = {
    'n_estimators': range(6, 10),
    'max_depth': range(3, 8),
    'learning_rate': [.001, .01, .02, .2, .3, .4],
    'colsample_bytree': [.7, .8, .9, 1]
}

# instanciation du classifier
xgb_c2 = XGBClassifier()

# gread search
g_search = GridSearchCV(estimator=xgb_c2, param_grid=param_grid,
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

""" excellent, on a atteint un accuracy de 100% grâce au GridSearch !!!! """

# matrice de confusion
cm = confusion_matrix(y_test, y_pred) 
print(cm)

