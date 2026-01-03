
![Project Logo](logo.png)

# Explainable Artificial Intelligence (XAI) - Project 2025-2026

## 1. Project Information
*   **Course:** Explainable Artificial Intelligence (XAI)
*   **Academic Year:** 2025–2026
*   **Professor:** Dr. S. MHAMMEDI
*   **Student:** Yassine Ben Acha
*   **Institution:** **Ecole Nationale de l'Intelligence Artificielle Berkane**

## 2. Project Description
This project focuses on the implementation, analysis, and critical evaluation of various Explainable AI (XAI) methods. The primary objective is to deconstruct "black box" machine learning models to understand their decision-making processes. Transparency and interpretability are crucial for building trust in AI systems, identifying biases, and complying with regulatory standards (e.g., GDPR).

The project covers a range of techniques from global explanations (understanding the model as a whole) to local explanations (understanding specific predictions), applied to diverse datasets.

## 3. Models Used
The following machine learning models were trained and interpreted:
*   **Support Vector Machine (SVM):** Used with RBF kernels to demonstrate non-linear decision boundaries and local approximations.
*   **Random Forest Classifier:** Used as a robust ensemble model to test feature importance and additive explanations.

## 4. Explainability Techniques Implemented
The project implements four major families of XAI methods:

*   **Global Methods:**
    *   **Partial Dependence Plots (PDP):** Visualizing the marginal effect of one or two features on the predicted outcome.
    *   **Accumulated Local Effects (ALE):** Isolating the effect of features while handling correlations better than PDP.

*   **Local Methods:**
    *   **SHAP (Shapley Additive Explanations):** 
        *   *Marginal SHAP:* Attribution based on marginal contributions.
        *   *KernelSHAP:* Model-agnostic approximation using weighted linear regression and sampling.
    *   **LIME (Local Interpretable Model-agnostic Explanations):** Approximating a complex model (SVM) locally with an interpretable surrogate (Decision Tree) using weighted sampling.
    *   **Counterfactual Explanations (What-If):** Identifying the minimal changes required in input features to flip the model's prediction.

## 5. Methodology
The project follows a structured approach for each technique:
1.  **Theoretical Implementation:** Several methods (e.g., KernelSHAP, LIME sampling/weighting, What-If) were implemented from scratch or low-level primitives to demonstrate understanding of the underlying algorithms.
2.  **Visualization:** Generating intuitive plots (decision surfaces, importance bars, grid visualizations) to interpret results.
3.  **Critical Analysis:** Answering specific interpretation questions to evaluate the reliability, validity, and limitations of each method.

**Key Strategies:**
*   **Sampling & Weighting:** For LIME and KernelSHAP, we implemented specific sampling strategies (uniform/perturbation) and exponential kernels for weighting based on proximity.
*   **Local Surrogates:** fitting simple models (Linear Regression, Decision Trees) on locally weighted data to derive explanations.

## 6. Experiments and Data
The techniques were applied to the following datasets:
*   **German Credit Dataset:** Used for PDP and ALE analysis (Exercise 1).
*   **FIFA Dataset:** Used for predicting "Man of the Match" and calculating SHAP values (Exercise 2).
*   **Wheat Seeds Dataset:** Used for classifying seed varieties with SVM (LIME) and Random Forest (Counterfactuals) (Exercises 3 & 4).

## 7. Key Findings
*   **Model Sensitivity:** Counterfactual analysis revealed that small changes in specific geometric features of seeds could alter classification, highlighting sensitivity in decision boundaries.
*   **Local vs. Global:** LIME visualizations demonstrated that global decision boundaries can be complex (non-linear), but local behavior can often be approximated linearly or by simple rules, though this approximation degrades with distance.
*   **Feature Importance:** SHAP provided a consistent ranking of features for individual predictions, distinguishing between factors that positively or negatively influenced the outcome.

## 8. Limitations identified
*   **Sampling Instability:** In LIME and KernelSHAP, the randomness in sampling can lead to variance in explanations across runs.
*   **Local Approximation:** Local surrogates are only valid within a small neighborhood; extrapolating them leads to errors.
*   **Correlation:** PDPs can be misleading when features are highly correlated, whereas ALE offers a more robust view in such cases.
*   **Minimality:** The "closest instance" counterfactual approach guarantees validity (real data points) but does not guarantee sparsity (minimal number of changed features).

## 9. How to Run the Project
The project is structured as a series of Jupyter Notebooks.

### Prerequisites
Ensure you have Python installed. You can install the required dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib shap lime ale-python gower ConfigSpace
```

*Note: `lime` and `whatif` modules are also provided as local Python files (`lime.py`, `whatif.py`) within the project structure.*

### Execution Order
Run the notebooks in the following order to reproduce the results:

1.  `exercise1_german_credit.ipynb` (if available) - PDP/ALE Analysis.
2.  `exercise2_shap.ipynb` - SHAP Implementation and Analysis.
3.  `exercise3_lime.ipynb` - LIME Implementation and Visualization.
4.  `exercise4_counterfactuals.ipynb` - Counterfactual/What-If Analysis.

## 10. License / Academic Notice
This code is provided for academic purposes as part of the XAI course evaluation. Plagiarism or unauthorized reproduction for other academic submissions is strictly prohibited.
