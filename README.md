📈 HR Analytics: Employee Attrition Prediction Project
This repository documents a complete data science pipeline for analyzing and predicting employee attrition (turnover) within an organization. The project moves from meticulous data cleaning through in-depth statistical analysis, culminating in the development and evaluation of machine learning models to identify employees most likely to leave. The ultimate goal is to provide actionable insights to HR executives for improving employee retention.

🛠 Project Stages
1. Data Cleaning and Preparation
The initial dataset was rigorously prepared to ensure its quality and fitness for modeling.

Deduplication: A crucial step involved identifying and removing 3,008 duplicate employee records to guarantee a dataset of unique individuals.

Missing Data: Verified that the dataset had zero missing values, simplifying the pipeline.

Final Data: The clean dataset consists of 11,991 unique employee records, ready for transformation.

2. Feature Engineering
Before training, the data was transformed to meet the requirements of the machine learning algorithms.

Encoding: Categorical features (like Departments and salary) were converted into a numerical format using techniques like Label Encoding and One-Hot Encoding.

Scaling: Quantitative features (e.g., satisfaction_level, average_montly_hours) were standardized using the StandardScaler to ensure all variables contribute equally to the model training process.

Handling Imbalance: The class imbalance inherent in the turnover target variable (left) was addressed using the Borderline-SMOTE oversampling technique to prevent the model from being biased toward the majority class (employees who stayed).

3. Exploratory Data Analysis (EDA) and Statistical Testing
This phase provided a deep understanding of the key drivers of employee turnover using Matplotlib and Seaborn for visualization.

Visual Analysis: Key trends were visualized, focusing on:

The relationship between Satisfaction Level and departure.

The influence of Workload (average_montly_hours, number_project) on attrition.

Turnover rates segmented by Departments and Salary level.

Hypothesis Testing: Formal statistical tests were conducted to confirm the significance of relationships:

Chi-squared test (chi2_contingency): Used to test for dependence between categorical variables (e.g., salary vs. left).

T-tests and ANOVA (ttest_ind, f_oneway): Used to compare the mean values of continuous features (e.g., time_spend_company) across different groups.

🚀 Model Building and Evaluation
A competition of various machine learning algorithms was performed to find the best predictive model.

Model Selection: The following classifiers were trained on the prepared data:

Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest Classifier

XGBoost Classifier (Gradient Boosting)

Optimization: GridSearchCV was utilized for systematic hyperparameter tuning, ensuring each model achieved its maximum performance potential.

Evaluation: Models were rigorously assessed using standard classification metrics:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

💻 Technologies and Libraries Used
Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning/Modeling: Scikit-learn (e.g., LogisticRegression, RandomForestClassifier, StandardScaler), XGBoost, Imbalanced-learn (BorderlineSMOTE)

Statistics: SciPy (chi2_contingency, ttest_ind, f_oneway)
