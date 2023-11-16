#============
# EXTREME GRADIENT BOOSTING POUR LA REGRESSION
# Objectif : expliquer et prédire les valeurs de plusieurs features
#============

#============ description des données
"""
	Le jeu de données insurance.csv contient des informations concernant des assurés et leurs frais de santé 
	(colonne expenses). L'objectif est de construire un modèle prédictif (regression linéaire multiple) 
	pour prédire ces frais pour mieux adapter le coût de l'assurance.


age: age of primary beneficiary
sex: insurance contractor gender, female, male
bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
    objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
children: Number of children covered by health insurance / Number of dependents
smoker: Smoking
region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
charges: Individual medical costs billed by health insurance
"""