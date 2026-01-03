
import json
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown: Header
nb.cells.append(nbf.v4.new_markdown_cell("""
# Exercice 2 (Partie B): Contre-factuels (What-If)

## Objectif
Implémenter l'approche What-If pour générer des contre-factuels pour un classifieur binaire.
L'idée est de trouver l'observation existante la plus proche dans le dataset qui a une prédiction différente de celle de l'instance d'intérêt.

Nous utilisons le fichier `whatif.py`.
"""))

# Code: Setup
nb.cells.append(nbf.v4.new_code_cell("""
import sys
sys.path.append(".")
import numpy as np
import pandas as pd
import gower
from sklearn import ensemble
from Dataset.dataset import Dataset
import whatif

# Setup Dataset (Wheat Seeds converted to Binary)
dataset = Dataset("wheat_seeds", range(0, 7), [7], normalize=True, categorical=True)

# Create binary classification task (Class 0 vs Rest)
y = dataset.y.copy()
# Remap: Class 0 remains 0, others become 1? Or just make it binary.
# The previous code did: y[y == 0] = 1. Let's check original classes.
# Original classes: 1, 2, 3. 
# Dataset.y usually straight from CSV. 
# If dataset is Wheat Seeds, types are 1, 2, 3.
# The code `y[y == 0] = 1` implies there's 0s.
# Let's trust the provided snippet logic, but maybe adjust for Wheat Seeds which is 1,2,3.
# We will use dataset.y directly and see.

# Re-read y to be safe
y = dataset.y.ravel()
# Make it binary: Class 1 vs not Class 1
y_binary = np.where(y == 1, 1, 0)

print("Classes distribution (Binary):", np.unique(y_binary, return_counts=True))

# Reserve first row for interest
X = dataset.X
x_interest = X[0,:].reshape(1, -1)

# Training set = Rest
X_train = np.delete(X, (0), axis=0)
y_train = np.delete(y_binary, (0), axis=0)

# Train Random Forest
model = ensemble.RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Probe x_interest
print("x_interest features:", x_interest)
print("Prediction x_interest:", model.predict(x_interest))
"""))

# Markdown: Part (a)
nb.cells.append(nbf.v4.new_markdown_cell("""
### (a) Génération et Test
"""))

# Code: Generate
nb.cells.append(nbf.v4.new_code_cell("""
# 1. Generate Counterfactual
print("Generating Counterfactual...")
cf = whatif.generate_whatif(x_interest=x_interest, model=model, dataset=X_train)

if cf is not None:
    print("Counterfactual found!")
    print("Comparison (Interest vs CF):")
    feature_names = dataset.get_input_labels()
    
    # Create DataFrame for nice display
    df_compare = pd.DataFrame(data=np.vstack([x_interest, cf]), columns=feature_names, index=["x_interest", "Counterfactual"])
    display(df_compare)
    
    # Show diff
    diff = cf - x_interest
    df_diff = pd.DataFrame(data=diff, columns=feature_names, index=["Diff"])
    display(df_diff)

    print("Prediction CF:", model.predict(cf))
else:
    print("No counterfactual found.")

# 2. Evaluate Minimality
if cf is not None:
    print("\\nEvaluating Minimality...")
    non_minimal = whatif.evaluate_counterfactual(counterfactual=cf, x_interest=x_interest, model=model)
    print("Indices of non-minimal features:", non_minimal)
    if non_minimal:
        print("Non-minimal features names:", [feature_names[i] for i in non_minimal])
"""))

# Markdown: Part (b) & Questions
nb.cells.append(nbf.v4.new_markdown_cell("""
### (b) Analyse des attributs et Interprétation

**Attributs satisfaits (Validité, Parcimonie, ...) :**

1.  **Validité** : Cette approche garantit la validité (le CF a bien la classe souhaitée) car elle sélectionne *par définition* une observation qui a une prédiction différente. Si une telle observation existe dans le dataset, la validité est assurée.
2.  **Plausibilité (Manifold Closeness)** : L'approche satisfait parfaitement la plausibilité car le contre-factuel est un *vrai* point de données existant. Il est donc réaliste et respecte les corrélations des données.
3.  **Parcimonie (Sparsity)** : Cette approche **ne garantit pas** la parcimonie. Comme on prend un point existant, il est probable que *toutes* les caractéristiques soient différentes de celles de $x_{interest}$, même légèrement. On ne minimise pas le nombre de changements (norme $L_0$), mais la distance globale (souvent Gower ou $L_1/L_2$).

**Avantages :**
- Validité garantie.
- Plausibilité parfaite (pas de point "hors distribution").
- Facile à implémenter.

**Inconvénients :**
- Pas de parcimonie (difficile de voir *quoi* changer précisément).
- Dépend de la taille et de la densité du dataset (si le dataset est clairsemé, le CF le plus proche peut être très loin).
- Confidentialité (révèle un vrai point de données du dataset d'entraînement).

**Questions d’interprétation :**

**1. Comment le point contre-factuel choisi modifie-t-il les caractéristiques par rapport à l’observation originale ?**
(Réponse dépendante de l'exécution, voir le tableau "Diff" ci-dessus). Le point choisi modifie probablement de multiples caractéristiques simultanément pour atterrir sur une observation existante d'une autre classe.

**2. Quelle(s) caractéristique(s) a/ont le plus changé pour atteindre la prédiction souhaitée ?**
En observant la ligne "Diff" (valeurs absolues), on peut identifier la caractéristique avec le plus grand delta. C'est souvent l'indicateur principal du changement de classe, bien que la distance de Gower normalise les différences.

**3. Que pouvez-vous conclure sur la sensibilité du modèle à certaines caractéristiques à partir du contre-factuel ?**
Si le contre-factuel le plus proche nécessite de grands changements sur certaines caractéristiques et peu sur d'autres, cela suggère que le modèle est peut-être plus sensible aux caractéristiques qui, une fois modifiées (même avec d'autres), suffisent à basculer la classe. Cependant, avec l'approche What-If (points réels), il est difficile d'isoler la sensibilité marginale d'une seule caractéristique car tout change en même temps. La présence de caractéristiques "non-minimales" (calculées en a) indique quelles caractéristiques n'étaient *pas* nécessaires pour le changement, affinant notre vue sur ce qui est important.
"""))

with open('exercise4_counterfactuals.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Created exercise4_counterfactuals.ipynb successfully.")
