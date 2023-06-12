#============
# ARBRES DE DECISIONS POUR LA CLASSIFICATION
# Objectif : prédire les valeurs d'une target binaire et expliquer l'impact des features sur la target
#============

#============ description des données
"""
On fait une étude sur le ronflement. Nous avons des caractéristiques physiologiques et d'habitudes de consommation de quelques patients.

caractéristiques physiologiques : taille, poids, age, sexe...
habitudes de consommation : alcool, fumeur...
Problématique : On veut savoir ce qui provoque le ronflement autrement dit, quelles sont les facteurs responsables du ronflement ?
"""

#============ vérifier la propreté du code
# pip install flake8P
# invoke flake8 (bash) : flake8

#============ chargement des bibliothèques
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#============ importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/ronfle.txt"
raw_df = pd.read_csv(path, sep = "\t")

#============ copie du dataset brut
mtcars_df = raw_df
mtcars_df.head()

#============ vérification des types
mtcars_df.dtypes

#============ afficher la dimension
print(mtcars_df.shape)

#============ matrice de correlation
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = mtcars_df.drop(['model'], axis=1).corr().round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Generate a mask for the upper triangle
plt.figure(figsize = (16,10))
sns.heatmap(corr_matrix, cmap = 'RdBu_r', mask = mask, annot = True)
plt.show()

"""
Problematique : quels sont les facteurs responsables du ronflement ?
"""

