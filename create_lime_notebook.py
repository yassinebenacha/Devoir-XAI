
import json
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown: Header
nb.cells.append(nbf.v4.new_markdown_cell("""
# Exercice 3 : LIME (SVM)

## Objectif
Implémenter LIME pour interpréter un Support Vector Machine (SVM) sur un problème de classification binaire (ou multiclasse) avec deux caractéristiques.

Nous utilisons le fichier `lime.py` qui contient l'implémentation des fonctions `sample_points`, `weight_points` et `fit_explainer_model`.
"""))

# Code: Imports and Setup
nb.cells.append(nbf.v4.new_code_cell("""
import sys
sys.path.append(".")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from Dataset.dataset import Dataset
import lime  # Corrected from 'limen'

# Load Dataset (Wheat Seeds)
dataset = Dataset("wheat_seeds", [1, 5], [7], normalize=True, categorical=True)
(X_train, y_train), (X_test, y_test) = dataset.get_data()

# Train SVM
model = SVC(gamma='auto')
model.fit(X_train, y_train)

# Define Interest Point and Parameters
x_interest = np.array([0.31, 0.37])
points_per_feature = 50
n_points = 1000
labels = dataset.get_input_labels()
colors = {0: "purple", 1: "green", 2: "orange"}
"""))

# Markdown: Part (a)
nb.cells.append(nbf.v4.new_markdown_cell("""
### (a) Inspection des fonctions implémentées

**1. Que représente la surface de prédiction affichée sur le graphique ?**
La surface de prédiction représente les frontières de décision globales du modèle SVM dans l'espace des deux caractéristiques sélectionnées. Les zones colorées indiquent la classe prédite par le modèle pour chaque point de cet espace.

**2. Que signifient les différentes couleurs observées dans l’espace des caractéristiques ?**
Les couleurs représentent les différentes classes cibles (variétés de blé dans ce dataset). Par exemple, violet, vert et orange correspondent aux trois classes possibles prédites par le modèle.

**3. Comment interprétez-vous les frontières de décision visibles sur le graphique ?**
Les frontières de décision séparent les zones de couleurs différentes. Elles indiquent où le modèle change de prédiction. Leur forme (courbée, non-linéaire) suggère que le noyau du SVM (probablement RBF) capture des relations non-linéaires complexes entre les caractéristiques.

**4. Que peut-on dire du comportement global du SVM à partir de cette visualisation ?**
On peut voir la capacité du SVM à séparer les classes. Certaines zones peuvent être complexes ou imbriquées, montrant où le modèle est incertain ou s'il fait du surapprentissage (si les frontières sont très irrégulières). Ici, les frontières semblent raisonnablement lisses.
"""))

# Code: Step 1 Get Grid
nb.cells.append(nbf.v4.new_code_cell("""
# 1. Get Grid and Plot SVM Decision Boundary
print("Running get_grid and plot_grid...")
u, v, z = lime.get_grid(model, dataset, points_per_feature=points_per_feature)
plt = lime.plot_grid(u, v, z, labels=labels, title="SVM")
plt.show()
"""))

# Markdown: Part (b)
nb.cells.append(nbf.v4.new_markdown_cell("""
### (b) Échantillonnage des points

**5. Pourquoi est-il nécessaire d’échantillonner des points autour de l’observation à expliquer ?**
Pour comprendre la décision du modèle *pour cette instance spécifique*, nous devons examiner comment le modèle se comporte dans son voisinage immédiat. L'échantillonnage génère des données de "sondage" pour capturer ce comportement local.

**6. Comment la taille de la zone échantillonnée influence-t-elle l’explication produite par LIME ?**
Une zone trop petite peut être bruitée ou ne pas capturer assez de variation pour déterminer une frontière locale (gradient instable). Une zone trop grande capture le comportement global du modèle, perdant la fidélité locale. Le défi est de trouver le bon équilibre pour une explication localement pertinente.

**7. Que se passe-t-il si les points sont échantillonnés trop loin du point à expliquer ?**
Les points éloignés reflètent la structure globale du modèle, qui peut être très différente de la structure locale autour de $x$. Si on les inclut avec trop d'importance, l'explication (le modèle linéaire ou l'arbre) essaiera d'approximer la frontière globale et échouera à expliquer pourquoi $x$ a été classé ainsi.
"""))

# Code: Step 2 Sample Points
nb.cells.append(nbf.v4.new_code_cell("""
# 2. Sample Points
print("Running sample_points...")
Z_sampled, y_sampled = lime.sample_points(model, dataset, n_points)

# Plot Sampled Points
plt = lime.plot_grid(u, v, z, labels=labels, title="SVM + Sampled Points")
lime.plot_points_in_grid(plt, Z_sampled, y_sampled, colors=colors)
plt.show()
"""))

# Markdown: Part (c)
nb.cells.append(nbf.v4.new_markdown_cell("""
### (c) Pondération des points

**8. Comment la distance au point x influence-t-elle le poids attribué à un point échantillonné ?**
La fonction de pondération (noyau exponentiel) attribue un poids élevé aux points proches de $x$ et un poids faible aux points éloignés. Le poids décroît rapidement à mesure que la distance augmente.

**9. Pourquoi les points proches de x ont-ils un poids plus élevé ?**
Car l'explication doit être *localement fidèle*. Nous voulons que le modèle explicatif (surrogate) reproduise très précisément le comportement du SVM pour les points qui ressemblent à $x$, car ce sont eux qui définissent la décision locale. Les points lointains sont moins pertinents.

**10. Comment cette pondération est-elle reflétée visuellement sur le graphique ?**
Sur les graphiques suivants, la taille des points sera proportionnelle à leur poids. Les points proches de l'instance d'intérêt (le point rouge) apparaîtront plus gros que les points éloignés.
"""))

# Code: Step 3 Weight Points
nb.cells.append(nbf.v4.new_code_cell("""
# 3. Weight Points
print("Running weight_points...")
weights = lime.weight_points(x_interest, Z_sampled)

# Plot Weighted Points
plt = lime.plot_grid(u, v, z, labels=labels, title="SVM + Weighted Sampled Points")
lime.plot_points_in_grid(plt, Z_sampled, y_sampled, weights, colors, x_interest)
plt.show()
"""))

# Markdown: Part (d)
nb.cells.append(nbf.v4.new_markdown_cell("""
### (d) Ajustement du modèle explicatif local (Arbre de Décision)

**11. Limitations potentielles/problèmes :**
Un arbre de décision est un modèle en escalier (non lisse). Il peut avoir du mal à approximer parfaitement une frontière de décision lisse ou oblique, même localement. De plus, il peut être instable (sensible aux faibles variations des données d'échantillonnage).

**12. Comment l’arbre de décision approxime-t-il localement la frontière de décision du SVM ?**
L'arbre partitionne l'espace local en rectangles homogènes. Ses frontières (lignes verticales/horizontales) tentent de s'aligner sur la frontière du SVM dans la zone de fort poids (près de $x$) pour minimiser l'erreur de classification pondérée.

**13. Dans quelles régions du graphique cette approximation est-elle la plus fiable ?**
L'approximation est la plus fiable dans le voisinage immédiat de $x$ (là où les points sont gros).

**14. Pourquoi l’approximation locale peut-elle devenir incorrecte loin du point x ?**
Loin de $x$, les poids sont faibles, donc l'arbre ne paie presque aucune pénalité pour faire des erreurs sur ces points. Il peut donc classer ces régions arbitrairement ou simplement étendre la classe majoritaire locale, ce qui ne correspond pas nécessairement à la réalité du SVM à cet endroit.

**15. Qu’est-ce que cela révèle sur la différence entre un modèle global et une explication locale ?**
Le modèle global (SVM) capture la complexité partout. L'explication locale (Arbre) est une simplification valable *uniquement* dans une petite fenêtre. On ne peut pas extrapoler l'explication locale pour comprendre le modèle dans son ensemble.
"""))

# Code: Step 4 Fit Explainer
nb.cells.append(nbf.v4.new_code_cell("""
# 4. Fit Explainer Model (Decision Tree)
print("Fitting explainer model...")
explainer = lime.fit_explainer_model(Z_sampled, y_sampled, weights)

# 5. Compare Models
fig = plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
lime.plot_grid(u, v, z, labels=labels, title="SVM (Global)", embedded=True)
lime.plot_points_in_grid(plt, Z_sampled, y_sampled, weights, colors, x_interest)

plt.subplot(1, 2, 2)
u2, v2, z2 = lime.get_grid(explainer, dataset, points_per_feature=points_per_feature)
lime.plot_grid(u2, v2, z2, labels=labels, title="Decision Tree (Local Surrogate)", embedded=True)
lime.plot_points_in_grid(plt, Z_sampled, y_sampled, weights, colors, x_interest)

plt.show()
"""))

# Markdown: Analysis
nb.cells.append(nbf.v4.new_markdown_cell("""
### Analyse critique basée sur les visualisations

**16. En quoi les graphiques générés par LIME facilitent-ils la compréhension du comportement du modèle ?**
Ils permettent de "voir" la décision : on voit où se situe l'instance par rapport à la frontière, quels voisins sont d'une autre classe, et la forme locale de la limite. Cela démystifie la "boîte noire".

**17. Dans quels cas les explications visuelles produites par LIME peuvent-elles être trompeuses ?**
Si on visualise seulement 2 dimensions alors que le modèle en utilise beaucoup plus, on peut rater des interactions critiques ("projection bias"). Aussi, si l'échantillonnage est insuffisant, la frontière locale affichée peut être un artefact du hasard et non la vraie frontière du modèle.

**18. Quel est l’impact du nombre de points échantillonnés sur la qualité de l’explication ?**
Plus de points = meilleure couverture de l'espace local = estimation plus stable et précise de la frontière locale. Trop peu de points = explication très variable et peu fiable.

**19. Comment la position du point x par rapport aux frontières de décision influence-t-elle l’interprétation ?**
Si $x$ est sur une frontière, l'interprétation montrera une forte sensibilité aux changements. Si $x$ est loin, l'interprétation dira "c'est robuste, rien à signaler". C'est crucial pour l'analyse de risque.

**20. Citez deux limites de la méthode LIME observables à partir des graphiques.**
1. **Instabilité** : Les points sont tirés au hasard. Si on relance, les points changent, et l'arbre explicatif pourrait changer légèrement.
2. **Définition du voisinage** : La décroissance des poids (taille des points) est continue, mais on ne sait pas objectivement où s'arrête le voisinage "pertinent". Le choix du `kernel_width` est subjectif et impacte fortement l'explication.
"""))

with open('exercise3_lime.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Created exercise3_lime.ipynb successfully.")
