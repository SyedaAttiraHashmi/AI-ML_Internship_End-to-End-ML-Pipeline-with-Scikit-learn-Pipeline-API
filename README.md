📊 Customer Churn Prediction – End-to-End ML Pipeline 🔍 Project Overview

Customer churn is a critical problem for subscription-based businesses. This project implements a production-ready, end-to-end machine learning pipeline to predict customer churn using the Telco Customer Churn dataset.

The solution covers the complete ML lifecycle:

Data preprocessing

Model training and tuning

Pipeline construction

Model persistence

Interactive prediction via Streamlit UI

🎯 Objective

Build a reusable and deployable machine learning pipeline for predicting customer churn using Scikit-learn’s Pipeline and ColumnTransformer APIs.

📁 Dataset

Telco Customer Churn Dataset

Customer demographics

Subscription services

Account and billing information

Target variable: Churn (Yes / No)

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Streamlit

Joblib

⚙️ Project Structure customer-churn-pipeline/ │ ├── churn_pipeline.pkl # Trained ML pipeline ├── app.py # Streamlit web application ├── train_pipeline.py # Model training & tuning script ├── Telco-Customer-Churn.csv # Dataset ├── requirements.txt └── README.md

🔄 Machine Learning Pipeline 🔹 Data Preprocessing

Numerical features scaled using StandardScaler

Categorical features encoded using OneHotEncoder

All preprocessing handled within a ColumnTransformer

🔹 Models Used

Logistic Regression

Random Forest Classifier

🔹 Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation

Best model selected based on accuracy

📈 Model Evaluation

Accuracy Score

Best models
best_logistic = log_grid.best_estimator_ best_rf = rf_grid.best_estimator_

Predictions
log_pred = best_logistic.predict(X_test) rf_pred = best_rf.predict(X_test)

Logistic Regression Accuracy: 0.8211497515968772 Random Forest Accuracy: 0.801277501774308

Classification Report

Random Forest Classification Report precision recall f1-score support

       0       0.84      0.90      0.87      1036
       1       0.66      0.51      0.58       373

accuracy                           0.80      1409
macro avg 0.75 0.71 0.72 1409 weighted avg 0.79 0.80 0.79 1409

Confusion Matrix Visualization alt text

💾 Model Export

The entire preprocessing + model pipeline is saved using joblib, ensuring:

No data leakage

Full reusability

Easy deployment

joblib.dump(best_model, "churn_model.pkl")

🖥️ Streamlit Web App

An interactive Streamlit UI allows users to:

Enter customer details

Predict churn probability

View real-time results

▶️ Run the App streamlit run app.py

📦 Installation pip install -r requirements.txt

🧠 Skills Demonstrated

End-to-end ML pipeline construction

Feature preprocessing using Pipeline API

Hyperparameter tuning with GridSearchCV

Model evaluation and persistence

Deployment-ready ML system

Streamlit UI development

alt text

📌 Key Highlights

✔ Production-ready ML workflow ✔ Single pipeline for preprocessing + prediction ✔ Reusable and scalable architecture ✔ Deployable via Streamlit

🚀 Future Enhancements

ROC-AUC & Precision-Recall metrics

SHAP-based model explainability

Batch CSV predictions

FastAPI deployment

Cloud deployment (Streamlit / AWS / GCP)
