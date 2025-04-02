# Customer Churn Prediction

## Project Overview
This project aims to predict customer churn using machine learning techniques. The dataset used is the "Telco Customer Churn" dataset, which contains various customer attributes such as contract type, tenure, and payment method. The goal is to identify customers who are likely to churn and help businesses take proactive measures.

## Dataset
- **File**: `Telco-Customer-Churn.csv`
- **Source**: A dataset containing customer demographics, service details, and churn status.
- **Target Variable**: `Churn` (Yes/No)

## Steps Involved
### 1. Data Preprocessing
- Handled missing values in the `TotalCharges` column by filling them with the mean value.
- Removed the `customerID` column as it is not useful for prediction.
- Converted categorical features into numerical format.

### 2. Exploratory Data Analysis (EDA)
- Visualized churn distribution using count plots and pie charts.
- Analyzed relationships between different features and churn rates.
- Identified that customers with month-to-month contracts have a higher churn rate.

### 3. Feature Engineering
- Encoded categorical variables using one-hot encoding.
- Scaled numerical features using MinMaxScaler.

### 4. Model Implementation
Implemented the following models:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Voting Classifier (Ensemble of multiple models)**

### 5. Model Evaluation
- Used metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Visualized model performance using confusion matrices and ROC curves.

## Results & Insights
- The **XGBoost classifier** performed the best, achieving the highest accuracy and ROC-AUC score.
- Customers with short-term contracts and lower tenure were more likely to churn.
- Payment methods and internet service type had a significant impact on churn rates.

## Installation & Usage
1. Clone this repository:
   ```sh
   git clone https://github.com/your-repo/churn-prediction.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```sh
   jupyter notebook "Customer Churn Prediction.ipynb"
   ```

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib & Seaborn (for visualization)

## Future Enhancements
- Implement deep learning models for better performance.
- Deploy the model as a web application.
- Improve feature selection and hyperparameter tuning.

## Author
Hamza Abbasi

