# 🏦 Loan Prediction using Machine Learning

This project predicts **loan approval eligibility** based on applicant details using **machine learning models**. It compares multiple algorithms and evaluates their performance on accuracy, recall, specificity, and precision.

---

## 🚀 Project Objectives

* Analyze a **loan approval dataset** through Exploratory Data Analysis (EDA).
* Build and evaluate machine learning models for predicting loan approval.
* Compare performance of:

  * Logistic Regression
  * Decision Tree
  * Random Forest
* Provide insights into important features influencing loan approval.

---

## 📊 Dataset

* Dataset: Public Loan Approval Dataset (Kaggle / UCI or academic source).
* Features include:

  * Applicant Income
  * Coapplicant Income
  * Loan Amount & Loan Term
  * Credit History
  * Gender, Marital Status, Dependents
  * Education, Employment Type
* Target Variable: **Loan_Status** (Approved / Not Approved).

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries Used:**

  * pandas, numpy (data preprocessing & analysis)
  * matplotlib, seaborn (visualization)
  * scikit-learn (ML models & evaluation)
  * Jupyter Notebook (development)

---

## ⚡ Project Workflow

1. **Data Collection** – Load dataset into Python.
2. **Data Cleaning** – Handle missing values, outliers, categorical encoding.
3. **Exploratory Data Analysis (EDA)** – Statistical insights & visualizations.
4. **Feature Engineering** – Create meaningful features for prediction.
5. **Model Training** – Train Logistic Regression, Decision Tree, and Random Forest.
6. **Model Evaluation** – Compare accuracy, recall, specificity, confusion matrix.
7. **Result Analysis** – Select best-performing model for deployment.

---

## 📂 Project Structure

```
loan-prediction/
│-- data/
│   └── loan_dataset.csv
│-- notebooks/
│   └── eda.ipynb
│   └── model_training.ipynb
│-- src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│-- results/
│   └── performance_report.csv
│-- README.md
```

---

## 🧪 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/loan-prediction.git
cd loan-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Jupyter Notebook or Python scripts:

```bash
jupyter notebook notebooks/eda.ipynb
```

4. Train models:

```bash
python src/train_model.py
```

---

## 🔮 Future Improvements

* Hyperparameter tuning for better accuracy.
* Try advanced models (XGBoost, LightGBM, Neural Networks).
* Build a **Flask/Django web app** for user-friendly loan prediction.
* Deploy model on **AWS / Heroku / College server**.

---

## 👩‍💻 Contributors

* **Sanika Dafal**
* **Amir Bensekar**

Guide: **Mrs. Manasi Shirurkar (IMCC Pune)**

---
