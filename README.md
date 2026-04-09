# 💳 Credit Risk Prediction - End-to-End ML Project

## 📌 Overview

This project implements an end-to-end Machine Learning system to predict **credit risk (default probability)** of clients based on financial and demographic data.

The goal is to simulate a real-world production scenario, including:

* Data processing pipelines
* Model training and evaluation
* Experiment tracking (MLflow)
* Model deployment via API (FastAPI)
* Reproducibility and clean architecture

---

## 🎯 Business Problem

Financial institutions need to assess whether a client is likely to **default on a loan**.

This model predicts:

* `0` → No default (low risk)
* `1` → Default (high risk)

---

## 🧠 Machine Learning Approach

* Problem type: **Binary Classification**
* Algorithms:

  * Logistic Regression (baseline)
  * Random Forest
  * XGBoost / LightGBM
* Evaluation metrics:

  * ROC-AUC
  * Precision / Recall
  * Confusion Matrix

---

## 🗂️ Project Structure

```
credit-risk-ml/

├── data/
│   ├── raw/                # Original dataset
│   ├── processed/          # Cleaned datasets
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │
│   ├── utils/
│
├── models/
│   ├── model.pkl
│
├── api/
│   ├── app.py              # FastAPI service
│
├── mlruns/                 # MLflow experiments
│
├── requirements.txt
├── .gitignore
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/ocovic/credit-risk-ml.git
cd credit-risk-ml

python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Train model

```bash
python src/models/train.py
```

### 3. Run API

```bash
uvicorn api.app:app --reload
```

---

## 🔌 API Example

### Request

```json
POST /predict

{
  "income": 1200,
  "age": 35,
  "debt_ratio": 0.4
}
```

### Response

```json
{
  "risk": "HIGH",
  "probability": 0.82
}
```

---

## 📊 MLOps Features

* MLflow experiment tracking
* Model versioning
* Reproducible pipelines
* Modular architecture

---

## 📈 Future Improvements

* Hyperparameter tuning
* SHAP interpretability
* Docker containerization
* CI/CD pipeline

---

## 👨‍💻 Author

Osvaldo Contreras
Software Engineer → Future ML Engineer 🚀

---
