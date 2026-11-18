# TalentPulse – Employee Attrition Prediction & Workforce Analytics

## 📌 Project Overview

**TalentPulse** is an end-to-end Machine Learning and Workforce Analytics project that analyzes employee attrition patterns and predicts whether an employee is likely to leave an organization.

The project combines data analysis, statistical exploration, feature engineering, machine learning, model evaluation, explainable AI, and SQL-based workforce analytics to identify factors associated with employee attrition.

The objective is to help HR teams understand workforce trends, identify potential retention risks, and make more informed, data-driven decisions.


---

## 🎯 Project Objectives

- Analyze employee demographics, job characteristics, and workforce patterns.
- Identify factors associated with employee attrition.
- Perform exploratory data analysis and statistical analysis.
- Build and compare multiple classification models.
- Handle class imbalance and evaluate different approaches.
- Tune model hyperparameters using GridSearchCV.
- Interpret model predictions using SHAP.
- Perform workforce analytics using MySQL.
- Develop an interactive application for attrition prediction.

---

## 📊 Dataset

**Dataset:** IBM HR Analytics Employee Attrition & Performance

**Dataset characteristics:**

| Description | Value |
|---|---:|
| Total Employees | 1,470 |
| Total Features | 35 |
| Employees Who Stayed | 1,233 |
| Employees Who Left | 237 |
| Attrition Rate | 16.12% |

**Target Variable:** `Attrition`

- `Yes` – Employee left the organization.
- `No` – Employee stayed with the organization.

The dataset contains employee information such as age, department, job role, monthly income, job satisfaction, overtime, work experience, and other workforce-related attributes.

---

## 🛠️ Technologies Used

### Programming and Data Analysis
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy

### Machine Learning
- Scikit-learn
- XGBoost
- Imbalanced-learn

### Model Explainability
- SHAP

### Database and SQL
- MySQL
- MySQL Workbench

### Deployment and Tools
- Streamlit
- JupyterLab
- Git
- GitHub
- Joblib

---

## 🔄 Project Workflow

### 1. Data Cleaning and Preparation

Performed data quality checks and prepared the dataset for analysis and modeling.

Activities included:

- Dataset inspection and structural analysis.
- Missing value checks.
- Duplicate record checks.
- Unique value and constant feature analysis.
- Identification of identifier and irrelevant columns.
- Data type and categorical value validation.
- Outlier assessment.
- Logical consistency checks.

Constant and identifier columns were excluded from model predictors where appropriate.

### 2. Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand employee characteristics and attrition patterns.

Analysis included:

- Overall attrition distribution.
- Department-wise attrition.
- Job-role-wise attrition.
- Overtime and attrition relationship.
- Salary and experience comparisons.
- Job satisfaction and work-life balance.
- Numerical feature distributions.
- Correlation analysis.

#### Key EDA Findings

| Observation | Result |
|---|---:|
| Overall Attrition Rate | 16.12% |
| Sales Department Attrition | 20.63% |
| HR Department Attrition | 19.05% |
| R&D Department Attrition | 13.84% |
| Sales Representative Attrition | 39.76% |
| Laboratory Technician Attrition | 23.94% |
| Attrition with Overtime | 30.53% |
| Attrition without Overtime | 10.44% |

Employees who left also showed lower average monthly income, total working experience, and tenure in their current roles compared with employees who stayed.

These findings represent associations in the dataset and should not be interpreted as proof of causation.

### 3. Statistical Analysis

Used statistical techniques to investigate differences between employee groups.

- Descriptive statistics.
- Correlation analysis.
- Mann–Whitney U test.
- Comparison of numerical features between employees who stayed and left.

For example, the Mann–Whitney U test was used to examine the difference in monthly income between the two attrition groups.

The analysis indicated a statistically significant difference in monthly income distributions.

### 4. Feature Engineering and Preprocessing

Prepared the dataset for machine learning.

Activities included:

- Separating independent variables and target variable.
- Removing identifier and constant features from model inputs.
- Identifying numerical and categorical features.
- Train-test splitting using stratification.
- StandardScaler for numerical features in applicable models.
- OneHotEncoder for categorical features.
- Building preprocessing pipelines using Scikit-learn.

The preprocessing steps were incorporated into pipelines to help prevent data leakage.

### 5. Handling Class Imbalance

The dataset contains significantly more employees who stayed than employees who left.

- No: 83.88%
- Yes: 16.12%

A baseline model predicting every employee as `No` would achieve approximately 83.88% accuracy but would fail to identify employees who leave.

To address this issue, different approaches were explored:

- Class weighting.
- `scale_pos_weight` in XGBoost.
- SMOTE.
- SMOTENC.

Resampling was applied to training data rather than the test data to avoid data leakage.

### 6. Model Development and Comparison

Multiple classification algorithms were implemented and evaluated.

Models explored:

1. Logistic Regression
2. Balanced Logistic Regression
3. Decision Tree
4. Random Forest
5. XGBoost

Hyperparameter optimization was explored using **GridSearchCV with cross-validation**, including experiments with Logistic Regression and XGBoost.

Model development also involved comparing class-weighting and resampling strategies.

#### Model Evaluation

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Since identifying employees at risk of leaving is important, particular attention was given to the precision, recall, and F1-score of the positive class (`Yes`).

#### Model Comparison

The following are the observed test-set results from the model experiments:

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1-score (Yes) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 86.05% | 61.54% | 34.04% | 43.84% | 0.812 |
| Balanced Logistic Regression | 75.17% | 34.88% | 63.83% | 45.11% | 0.803 |
| Decision Tree | 77.89% | 29% | 25.53% | 27% | 0.567 |
| Random Forest | 84.35% | 56% | 10.64% | 18% | 0.788 |
| Original XGBoost | 82.65% | 46.15% | 51.06% | 48.48% | 0.789 |

**Note:** These are observed results from the reported model experiments. Results from resampling and hyperparameter-tuning experiments were also examined. Metrics can vary with model configuration and evaluation setup.

### 7. Final Model – XGBoost

The original XGBoost model was selected as the final model based on its positive-class F1-score and balance between precision and recall among the reported experiments.

The model uses:

- XGBoost Classifier.
- Tree-based preprocessing pipeline.
- One-hot encoding for categorical features.
- Class imbalance weighting using `scale_pos_weight`.
- Regularization.
- Hyperparameter configuration to control model complexity.

#### Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 82.65% |
| Precision (Yes) | 46.15% |
| Recall (Yes) | 51.06% |
| F1-score (Yes) | 48.48% |
| ROC-AUC | 0.789 |

#### Confusion Matrix

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 219 | 28 |
| Actual Yes | 23 | 24 |

The model correctly identified 24 out of 47 employees who left in the test set.

It also identified some employees as potential attrition risks who did not leave. This highlights the trade-off between precision and recall.

### 8. Model Explainability – SHAP

SHAP (SHapley Additive exPlanations) was used to understand how different features influence the model's predictions.

The analysis helps explain:

- Which features contribute most to model predictions.
- How individual feature values influence predictions.
- Which characteristics increase or decrease the model's estimated attrition risk.

SHAP provides explanations of model behavior, not causal relationships.

### 9. SQL-Based Workforce Analytics

SQL analysis was performed separately using **MySQL Workbench** after importing the employee dataset into a MySQL database.

The database was named `TalentPulse`, with an employee attrition table.

SQL was used to answer workforce-related business questions, including:

1. What is the overall employee attrition rate?
2. Which departments have higher attrition?
3. Which job roles experience higher attrition?
4. How does overtime relate to employee attrition?
5. How do salary and work experience differ between employees who stayed and left?

SQL concepts used included:

- SELECT
- WHERE
- GROUP BY
- ORDER BY
- CASE
- Aggregate functions
- Conditional aggregation
- Subqueries

**Important:** SQL queries were executed in MySQL Workbench, not directly inside the Jupyter notebook. The notebook documents the SQL analysis and its findings.

### 10. Business Insights and Recommendations

Based on the analysis, the following observations and recommendations were identified.

| Finding | Recommendation |
|---|---|
| Higher attrition among Sales Representatives | Review role-specific workload, incentives, and career progression. |
| Higher attrition among employees working overtime | Investigate workload distribution and work-life balance. |
| Lower average tenure among employees who left | Strengthen onboarding, mentoring, and early-career support. |
| Differences in monthly income between groups | Review compensation and internal pay equity. |
| Attrition varies across departments | Develop targeted retention strategies rather than applying one approach to all departments. |

These recommendations are intended to support HR analysis and should be validated using additional organizational data.

### 11. Model Saving and Deployment

The trained XGBoost pipeline was saved using Joblib.

The project includes a Streamlit application designed to:

- Accept employee information through an interactive form.
- Load the saved machine learning pipeline.
- Predict the employee's attrition class.
- Display the estimated attrition probability.

The application is intended as a demonstration of the model's capabilities.

Predictions are estimates and should not be used as the sole basis for employment-related decisions.

---

## 📁 Project Structure

```text
TalentPulse/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│       └── cleaned_employee_data.csv
│
├── notebooks/
│   └── TalentPulse_Employee_Attrition.ipynb
│
├── models/
│   └── talentpulse_xgboost.pkl

```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/TalentPulse.git
```

### 2. Navigate to the Project Directory

```bash
cd TalentPulse
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available at:

```text
https://talentpulse-employee-attrition-hrintelligence.streamlit.app/
```

---

## 🚀 Future Improvements

- Improve minority-class prediction performance.
- Evaluate decision thresholds using cross-validation.
- Validate the model on an independent dataset.
- Add more workforce analytics and interactive visualizations.
- Improve model explainability through individual employee-level SHAP explanations.
- Integrate a database-backed analytics workflow.
- Deploy the application publicly.

---

## ⚠️ Limitations

- The project uses a public IBM HR Analytics dataset, not real company employee data.
- The dataset may not represent current workforce patterns in every organization.
- Model predictions are probabilistic estimates, not guaranteed outcomes.
- Relationships identified through EDA and SHAP should not be interpreted as causal.
- Model performance may change when applied to different datasets or organizations.
- The model should support human decision-making, not replace it.

---

## 👨‍💻 Author

**Shiva Kumar Chennaboina**
**sky10shiva@gmail.com**


## ⭐ Acknowledgment

Dataset: IBM HR Analytics Employee Attrition & Performance dataset, publicly available for educational and analytical use.
