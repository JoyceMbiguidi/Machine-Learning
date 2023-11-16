#============
# K NEAREST NEIGHBORS POUR LA REGRESSION
# Objectif : expliquer et prédire les valeurs d'une feature
#============


#============ description des données
"""
    A data frame with 935 observations on 17 variables:
        - wage: monthly earnings
        - hours: average weekly hours
        - IQ: IQ score
        - KWW: knowledge of world work score
        - educ: years of education
        - exper: years of work experience
        - tenure: years with current employer
        - age: age in years
        - black: 1 if black
        - married: 1 if married
        - meduc: mother's education
        - feduc: father's education
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


#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/wage.csv"
raw_df = pd.read_csv(path, sep = ",", decimal = ".",)

#============ copie du dataset brut
wage_df = raw_df
wage_df.head()

#============ data wrangling
wage_df.isna().sum() 

print("feduc contient {} valeurs manquantes.".format(wage_df['feduc'].isna().sum()), "\n", 
      "soit {:.2%} du jeu de données".format(wage_df['feduc'].isna().sum() / len(wage_df)))


#============ dataviz
sns.kdeplot(wage_df['feduc'], shade=True)
plt.xlabel("Feduc")  # Set the x-axis label
plt.ylabel("Density")  # Set the y-axis label
plt.title("Density Plot of Feduc")  # Set the plot title
plt.axvline(np.mean(wage_df['feduc']), color='red', linestyle='dashed', linewidth=1, label='Average')
plt.axvline(np.std(wage_df['feduc']), color='green', linestyle='dashed', linewidth=1, label='Standard deviation')
plt.legend()

# Show the plot
plt.show()

# on peut donc imputer FEDUC par la moyenne


sns.kdeplot(wage_df['meduc'], shade=True)
plt.xlabel("Meduc")  # Set the x-axis label
plt.ylabel("Density")  # Set the y-axis label
plt.title("Density Plot of Meduc")  # Set the plot title
plt.axvline(np.mean(wage_df['meduc']), color='red', linestyle='dashed', linewidth=1, label='Average')
plt.axvline(np.std(wage_df['meduc']), color='green', linestyle='dashed', linewidth=1, label='Standard deviation')
plt.legend()

# Show the plot
plt.show()

# on peut donc imputer MEDUC par la moyenne ou la médiane


#============ imputation des valeurs manquantes
wage_df['feduc'].fillna(wage_df['feduc'].median(),axis=0, inplace=True )
#wage_df.drop(['feduc'], axis=1)
wage_df['meduc'].fillna(wage_df['meduc'].median(),axis=0, inplace=True )

wage_df.info()


#============ matrice de correlation
mask = np.triu(np.ones_like(wage_df.corr(), dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(wage_df.corr(), cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()


#============ scaling
scaler=StandardScaler()
wage_df_sc= scaler.fit_transform(wage_df)

wage_df_sc = pd.DataFrame(wage_df_sc, columns=wage_df.columns)
wage_df_sc.head()

wage_df_sc['married']=wage_df['married']
wage_df_sc['black']=wage_df['black']
wage_df_sc.head()

"""
Problematique : Qu'est ce qui influence le niveau de salaire ?
"""

#============ variable explicative
x = wage_df_sc.drop(['wage'], axis = 1)

#============ variable à expliquer
y = wage_df_sc['wage']

#============ séparation des données : train - test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)


#============ Modele 1
from sklearn.neighbors import KNeighborsRegressor
# entrainement du modele
KNN_regression = KNeighborsRegressor(n_neighbors=5)
KNN_regression.fit(x_train, y_train)

# prédiction sur le jeu de test
y_test_predict = KNN_regression.predict(x_test)
predictions = pd.DataFrame({ 'y_test':y_test,'y_predict':y_test_predict})
predictions.head()


# évaluation du modèle sur l'ensemble de données de test
sns.scatterplot(x=y_test, y=y_test_predict, alpha=0.6, size=y_test_predict, hue=y_test_predict)
sns.regplot(x=y_test, y=y_test_predict, scatter=False, color='orange', label="Regression Line")

plt.xlabel('wage réel', fontsize=14)
plt.ylabel('wage prédit', fontsize=14)
plt.title('wage réel vs wage prédit (jeu de test)', fontsize=17)

plt.legend()

plt.show()


#============ métriques : coefficient de détermination
R2_train = KNN_regression.score(x_train, y_train)
R2_test = KNN_regression.score(x_test, y_test)
MSE_test = round(np.mean(np.square(y_test - y_test_predict)), 2)
RMSE_test = round(np.sqrt(MSE_test), 2)


print("-----------------------------------------")
print('MSE_test  = {}'.format(MSE_test))
print('RMSE_test  = {}'.format(RMSE_test))
print('R2_train = {}'.format(R2_train))
print('R2_test = {}'.format(R2_test))
print("-----------------------------------------")



#============ validation croisée : objectif, trouver l'optimum K
from sklearn.model_selection import cross_val_score
NMSE = cross_val_score(estimator = KNN_regression, X = x_train, y = y_train, cv = 5)
MSE_CV = round(np.mean(-NMSE),4)
MSE_CV


#============ choix du K
RMSE_CV=[]
RMSE_test = []

k=40

for i in range(1,k):
    KNN_i = KNeighborsRegressor(n_neighbors=i)
    KNN_i.fit(x_train, y_train)
    RMSE_i = np.sqrt(np.mean(-1*cross_val_score(estimator = KNN_i, X = x_train, y = y_train, cv = 10)))
    RMSE_CV.append(RMSE_i)
    
    RMSE_test.append(np.sqrt(np.mean(np.square(y_test - KNN_i.predict(x_test)))))
    
optimal_k = pd.DataFrame({'RMSE_CV': np.round(RMSE_CV,2), 'RMSE_test':np.round(RMSE_test,2)}, index=range(1,k))


optimal_k.head(10)

np.argmin(optimal_k['RMSE_CV'])


#============ dataviz
plt.figure(figsize=(10,5))
sns.lineplot(data=optimal_k)
plt.title('Validation croisée RMSE VS K')
plt.xlabel('K')
plt.ylabel('RMSE')
plt.show()











































































