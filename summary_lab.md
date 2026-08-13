# Laboratory Summary: Machine Learning for Concrete Carbonation Depth Prediction

## 1. Domain Context & Objective
- **Initial Architecture:** The original approach attempted to predict both **Carbonation Depth** and **Compressive Strength** as dual outputs.
- **Expert Revision:** After consulting with concrete domain specialists, it was concluded that predicting compressive strength alongside carbonation depth is physically unrepresentative for this use case. The correlation between Compressive strength and Recycled aggregate is inverse and non-linear.
- **Final Architecture:** **Compressive Strength** was shifted to be an **input feature**. The models now focus purely on a single-target regression: **Carbonation Depth (mm)**.
- **Monte Carlo:** The multi-objective Monte Carlo simulation was removed from the application as the system was reduced to a single target.

## 2. Dataset Processing
- **File Name:** `cube_strength_carbonation_depth_data.csv`
- **Samples:** 529 experimental records.
- **Input Features (9):**
  1. Water absorption (%)
  2. Effective w/b ratio
  3. Fine aggregate (kg/m³)
  4. Gravel content (kg/m³)
  5. RA content (kg/m³) - Recycled Aggregate
  6. Superplasticizer (kg/m³)
  7. CO2 concentration (%)
  8. Exposure time (days)
  9. Compressive strength (MPa)
- **Target Output (1):**
  1. Carbonation Depth (mm)

## 3. Machine Learning Pipeline
### Training Methodology
- **Validation Strategy:** 5-Fold Cross-Validation.
- **Hyperparameter Tuning:** Automated via **Optuna** (Bayesian Optimization) for tree-based and linear models.
- **Preprocessing:** Inputs were scaled using standard scalers (`StandardScaler`) fitted inside each fold to prevent data leakage.

### Hyperparameter Search Spaces (Optuna)
| Model | Hyperparameters Tuned | Range / Search Space |
|-------|-----------------------|----------------------|
| **XGBoost** | `n_estimators`, `max_depth`, `learning_rate` | [50 - 300], [3 - 10], [1e-3 - 0.3 (log)] |
| **LightGBM** | `n_estimators`, `max_depth`, `learning_rate`, `num_leaves` | [50 - 300], [3 - 15], [1e-3 - 0.3 (log)], [15 - 255] |
| **CatBoost** | `iterations`, `depth`, `learning_rate` | [50 - 300], [3 - 10], [1e-3 - 0.3 (log)] |
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split` | [50 - 300], [3 - 20], [2 - 10] |
| **SVR** | `C`, `epsilon`, `gamma` | [1e-3 - 1e3 (log)], [1e-4 - 1.0 (log)], [1e-4 - 1.0 (log)] |
| **Ridge Regression** | `alpha` | [1e-3 - 1e3 (log)] |
| **TabNet** | `n_da`, `n_steps`, `gamma`, `lambda_sparse` | [8 - 32], [3 - 7], [1.0 - 1.8], [1e-4 - 1e-1 (log)] |

### Performance Metrics Evaluation
*Note: Evaluated across the full dataset using the trained predictors (ensemble of 5 folds).*

| Model | MSE | RMSE (mm) | MAE (mm) | R² Score |
|-------|-----|-----------|----------|----------|
| **XGBoost** | **0.3900** | **0.6245** | **0.4330** | **0.9953** |
| **CatBoost** | 0.4063 | 0.6374 | 0.4480 | 0.9951 |
| **LightGBM** | 1.0051 | 1.0026 | 0.6210 | 0.9879 |
| **Random Forest** | 1.5534 | 1.2464 | 0.8322 | 0.9812 |
| **SVR** | 2.8752 | 1.6956 | 1.0081 | 0.9653 |
| **MLP (Deep Learning)** | 6.1046 | 2.4708 | 1.7891 | 0.9263 |
| **Ridge Regression** | 50.2644 | 7.0897 | 4.9055 | 0.3930 |
| **TabNet** | 193.3912 | 13.9065 | 10.7782 | -1.3355 |

*Conclusion: Tree-based ensemble models (XGBoost, CatBoost) dramatically outperformed linear models and neural networks (MLP, TabNet) for this tabular dataset.*

## 4. Interpretability (SHAP Analysis)
To ensure the machine learning models were physically sound and transparent, **SHAP (SHapley Additive exPlanations)** was implemented using `shap.TreeExplainer`.
- **Global Interpretability:**
  - **Beeswarm Plot:** Extracted feature directionality and impact magnitude. Showed non-linear dependencies.
  - **Bar Chart:** Ranked features by absolute mean SHAP value to determine global feature importance.
- **Local Interpretability:**
  - **Waterfall Plot:** Deconstructs a single prediction instance, explaining exactly how much each feature contributed (in mm) to push the carbonation depth prediction away from the baseline.

## 5. Deployment / Web Application
- Built a **Streamlit** dashboard implementing a Clean Architecture pattern.
- **Interactive Inputs:** Sidebar contains bounded sliders for all 9 input features (including Compressive Strength).
- **Tabs:**
  - **Interactive Predictions:** Side-by-side metric comparison across all 8 models and a dynamic SHAP Waterfall chart.
  - **Model Performance:** Cross-validation metrics table.
  - **Learning Curves:** Train vs Validation loss history for deep learning and boosting models.
  - **SHAP Interpretability:** Global static plots (Beeswarm, Bar) for model introspection.
