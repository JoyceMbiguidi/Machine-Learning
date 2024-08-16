# Importation des bibliothèques
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Importation des données
path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/iris.csv"
iris_df = pd.read_csv(path, sep = ";", encoding = 'utf-8')
""" le séprateur par défaut est le point virgule et l'encodage du fichier UTF-8"""

# Affichage des premières lignes
iris_df.head()

# Imputation des valeurs manquantes
iris_df.fillna(iris_df.drop("Species", axis = 1).mean(), inplace=True)
""" ne jamais imputer des valeurs manquantes de façon arbitraire et sans vérifier le type de chaque colonne."""

# Séparation des features et de la cible
X = iris_df.drop("Species", axis=1)
y = iris_df["Species"]
""" Species doit correctement être orthographié. Les données auraient pu être standardisées et bravo si vous l'avez proposé !"""

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
""" un split 50-50 était utilisé, ce qui n'est pas idéal car une proportion plus faible pour le test (20-30%) est généralement préférable."""

# Construction du modèle
model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation du modèle avec une explication inadéquate des métriques
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Précision du modèle: {accuracy * 100:.0f}%")
print(f"Matrice de confusion :\n{conf_matrix}")
print(f"Le modèle a une accuracy de {accuracy * 100:.0f}%, ce qui signifie que, dans {accuracy * 100:.0f}% des cas, il prédit correctement la classe de l'exemple.\nBien que ce soit un bon indicateur général de performance, il est important de ne pas se fier uniquement à cette métrique.\nL'accuracy peut être trompeuse, surtout dans le cas de classes déséquilibrées.\nPour évaluer plus précisément les performances du modèle, il est crucial d'examiner d'autres métriques comme la précision, le rappel et le F1-score pour chaque classe.\nCes métriques nous permettent de comprendre si le modèle est biaisé vers certaines classes et comment il se comporte dans des situations où certaines classes sont minoritaires.\nEn outre, il est important d'examiner la matrice de confusion pour voir comment les prédictions sont distribuées entre les classes.\nPar exemple, un modèle pourrait avoir une bonne accuracy globale tout en étant moins performant sur des classes minoritaires, ce qui peut être problématique selon le contexte d'utilisation.")