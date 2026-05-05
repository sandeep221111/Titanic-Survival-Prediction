#  Titanic Survival Prediction using Logistic Regression

##  Project Overview

This project aims to predict whether a passenger survived the Titanic disaster using machine learning. It is a **binary classification problem**, where the target variable is `Survived` (0 = No, 1 = Yes).

---

##  Dataset

* Dataset used: Titanic dataset (from Kaggle)
* Contains information about passengers such as age, gender, ticket class, fare, etc.

---

##  Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

##  Steps Performed

### 1. Data Loading

* Loaded dataset using Pandas

### 2. Data Cleaning

* Filled missing values:

  * `Age` → median
  * `Embarked` → mode
* Dropped `Cabin` due to excessive missing values

### 3. Feature Engineering

* Created new features:

  * `FamilySize` = SibSp + Parch + 1
  * `IsAlone` = whether passenger is alone

### 4. Data Preprocessing

* Converted categorical features:

  * `Sex` → numerical
  * `Embarked` → one-hot encoding
* Dropped irrelevant columns:

  * `Name`, `Ticket`, `PassengerId`

### 5. Model Building

* Used **Logistic Regression**
* Applied **StandardScaler** for feature scaling
* Split data into train and test sets (80-20)

### 6. Model Evaluation

* Accuracy: **~80%**
* Used:

  * Confusion Matrix
  * Classification Report

---

##  Results

* Achieved ~80% accuracy on test data
* Model performs slightly better in predicting non-survivors compared to survivors

---

##  How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the project
python3 titanic.py
```

---

##  Project Structure

```
titanic-project/
│── titanic.py
│── Titanic-Dataset.csv
│── README.md
```

---

##  Future Improvements

* Try advanced models (Random Forest, XGBoost)
* Hyperparameter tuning
* Deploy model using Flask / Streamlit

---

##  Author

Sandeep Sharma
