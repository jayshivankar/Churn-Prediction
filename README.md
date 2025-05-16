# 🧠 Customer Churn Prediction using Deep Learning (TensorFlow & Keras)

This project uses a simple deep learning model built with TensorFlow/Keras to predict customer churn based on structured customer data.

---

## 📘 Project Overview

Customer churn prediction helps businesses proactively identify customers who are likely to leave, enabling them to take targeted retention measures. This notebook performs:

- Data preprocessing and cleaning  
- Feature encoding and scaling  
- Deep learning model training using Keras  
- Evaluation of model performance  

---

## 📂 Files

- `churn_prediction.ipynb` — Jupyter Notebook with the full pipeline  
- `customer_churn.csv` — Dataset used for training  
- `README.md` — Project description and instructions  

---

## 📊 Dataset

The dataset includes customer information such as:

- Demographic info: `Gender`, `Age`, etc.  
- Account features: `Tenure`, `MonthlyCharges`, etc.  
- Service-related fields: `InternetService`, `Contract`, etc.  
- **Target**: `Churn` (binary)  

---

## ⚙️ Model Architecture

A simple feedforward neural network with:

- **Input Layer**: 26 features  
- **Hidden Layer**: 15 units with ReLU activation  
- **Output Layer**: 1 unit with Sigmoid activation for binary classification  

```python
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
