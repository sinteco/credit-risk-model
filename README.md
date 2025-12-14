# Credit Risk Scoring Model

## Project Overview
This project aims to create a Credit Scoring Model for a buy-now-pay-later service in partnership with an eCommerce company. The goal is to estimate the likelihood of default for potential borrowers using behavioral data.

## Project Structure
- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for EDA and analysis.
- `src/`: Source code for data processing, training, and inference.
- `api/`: FastAPI application for serving the model.
- `tests/`: Unit tests.

## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability
The Basel II Capital Accord emphasizes the need for rigorous risk measurement to determine capital requirements. It requires financial institutions to have a robust internal rating system for credit risk. This implies that credit scoring models must be:
- **Interpretable:** The logic behind the score must be understandable to regulators and auditors.
- **Well-Documented:** The development, validation, and operation of the model must be thoroughly documented to ensure transparency and reproducibility.
- **Validatable:** The model's performance must be regularly monitored and validated against actual outcomes.
A "black box" model that cannot be explained or audited would likely fail to meet these regulatory standards.

### 2. Proxy Variable for Default
In the absence of a direct "default" label (e.g., historical loan repayment records), we must create a proxy variable to represent credit risk. This is often done by analyzing behavioral data such as Recency, Frequency, and Monetary (RFM) patterns.
**Necessity:** Without a target variable, supervised learning is impossible. The proxy serves as the ground truth for training the model.
**Business Risks:**
- **Misclassification:** The proxy might not perfectly correlate with actual creditworthiness. A customer with low RFM scores might simply be a new user, not a high-risk one (Type I error), leading to lost revenue. Conversely, a high-spending user might still default (Type II error), leading to financial loss.
- **Bias:** The proxy might inadvertently encode biases present in the behavioral data, leading to unfair scoring.

### 3. Trade-offs: Logistic Regression (WoE) vs. Gradient Boosting
| Feature | Logistic Regression with WoE | Gradient Boosting (e.g., XGBoost, LightGBM) |
| :--- | :--- | :--- |
| **Interpretability** | **High.** Coefficients directly relate to the Weight of Evidence (WoE) of features, making it easy to explain *why* a score was assigned. | **Low.** Complex ensemble of trees makes it difficult to trace the decision path for individual predictions ("Black Box"). |
| **Performance** | **Moderate.** Assumes linear relationships (after transformation). May miss complex, non-linear interactions. | **High.** Can capture complex, non-linear patterns and interactions between features, often resulting in better predictive accuracy. |
| **Regulatory Compliance** | **Easier.** Widely accepted and understood by regulators. | **Harder.** Requires advanced explainability techniques (e.g., SHAP values) to justify decisions to regulators. |
| **Implementation** | **Simple.** Easy to implement and deploy. | **Complex.** Requires more tuning and computational resources. |

In a regulated financial context, the trade-off often leans towards **Logistic Regression** for its transparency, unless the performance gain from Gradient Boosting is significant and can be sufficiently explained using interpretability tools.

## Task 2: Exploratory Data Analysis (EDA)
**Objective:** Explore the dataset to uncover patterns, identify data quality issues, and form hypotheses that will guide feature engineering.

**Key Activities:**
- **Overview of the Data:** Understanding structure, rows, columns, and data types.
- **Summary Statistics:** Analyzing central tendency, dispersion, and shape of distributions.
- **Distribution Analysis:** Visualizing numerical and categorical features.
- **Correlation Analysis:** Understanding relationships between numerical features.
- **Missing Values & Outliers:** Identifying data quality issues.

**Deliverables:**
- `notebooks/eda.ipynb`: Jupyter notebook containing all EDA code and visualizations.
- Summary of top 3–5 most important insights.

## Task 3: Feature Engineering
**Objective:** Build a robust, automated, and reproducible data processing script that transforms raw data into a model-ready format.

**Key Activities:**
- **Pipeline Construction:** Using `sklearn.pipeline.Pipeline` to chain transformation steps.
- **Feature Engineering:** Creating aggregate features and interaction terms.
- **Handling Missing Values:** Imputation strategies (Median/Mode).
- **Encoding:** One-Hot Encoding for categorical variables.
- **Scaling:** Standardization of numerical features.
- **WoE and IV:** Implementing Weight of Evidence and Information Value calculations for feature selection and transformation.

**Deliverables:**
- `src/data_processing.py`: Python script containing the feature engineering pipeline and WoE implementation.

## Task 4: Proxy Target Variable Engineering
**Objective:** Create a credit risk target variable (`is_high_risk`) since the dataset lacks a pre-existing label.

**Key Activities:**
- **RFM Calculation:** Calculating Recency, Frequency, and Monetary metrics for each customer.
- **Clustering:** Using K-Means to segment customers into 3 groups based on RFM profiles.
- **Risk Labeling:** Identifying the "disengaged" (low frequency, low monetary) cluster and labeling it as high-risk (1).
- **Integration:** Merging the new target variable back into the dataset.

**Deliverables:**
- `src/proxy_labeling.py`: Python module for RFM calculation and risk labeling.
- `notebooks/proxy_labeling.ipynb`: Notebook demonstrating the process with synthetic data (due to dataset limitations).

## Task 5: Model Training and Tracking
**Objective:** Develop a structured model training process that includes experiment tracking, model versioning, and unit testing.

**Key Activities:**
- **Model Selection:** Training Logistic Regression and Random Forest models.
- **Hyperparameter Tuning:** Using `GridSearchCV` to optimize model performance.
- **Experiment Tracking:** Logging parameters, metrics, and artifacts to MLflow.
- **Evaluation:** Assessing models using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- **Unit Testing:** Writing tests for data processing functions using `pytest`.

**Deliverables:**
- `src/train.py`: Script for training, tuning, and logging models.
- `tests/test_data_processing.py`: Unit tests for the data pipeline.

### Experiment Results
| Model | Accuracy | ROC-AUC | Best Parameters |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 91.4% | 0.959 | `{'C': 10.0, 'solver': 'lbfgs'}` |
| **Random Forest** | 94.4% | 0.956 | `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}` |

**Selected Model:** Logistic Regression (Highest ROC-AUC).

### Test Results
All unit tests passed successfully:
```bash
tests/test_data_processing.py ..                                                              [100%]
========================================= 2 passed in 0.66s =========================================
```
