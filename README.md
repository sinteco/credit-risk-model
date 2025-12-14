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
