# 💊 Efficient Drug Discovery using Molecular Data

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?style=for-the-badge&logo=keras)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

> A deep learning-powered web application that classifies drug candidates as **viable or non-viable** using molecular descriptor data — built to accelerate early-stage drug screening.

---

## 🧬 Project Overview

Drug discovery is one of the most expensive and time-consuming processes in pharmaceutical research. This project applies machine learning and deep neural networks to **predict whether a drug candidate is biologically active** based on its molecular fingerprint/descriptors, enabling faster and cheaper pre-screening.

A fully interactive **Streamlit web app** allows users to input molecular features and get real-time predictions with confidence scores.

---

## ⚙️ Features

- ✅ End-to-end ML pipeline: data ingestion → preprocessing → model training → inference
- ✅ Deep Neural Network built with TensorFlow/Keras
- ✅ Feature engineering on molecular descriptors (RDKit-based features)
- ✅ Hyperparameter tuning for optimal model performance
- ✅ Model evaluation: Precision, Recall, F1-Score, ROC-AUC
- ✅ Deployed via Streamlit for real-time drug candidate screening

---

## 🗂️ Project Structure

```
drug-discovery/
│
├── data/
│   ├── raw/                   # Raw molecular dataset
│   └── processed/             # Cleaned & feature-engineered data
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Feature engineering & data cleaning
│   └── 03_modeling.ipynb      # Model training & evaluation
│
├── src/
│   ├── preprocess.py          # Data preprocessing pipeline
│   ├── model.py               # DNN architecture definition
│   ├── train.py               # Training script
│   └── predict.py             # Inference utility
│
├── app/
│   └── streamlit_app.py       # Streamlit web application
│
├── models/
│   └── best_model.h5          # Saved trained model
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

```
Input Layer  →  [Molecular Descriptors]
     ↓
Dense(256, ReLU) + BatchNorm + Dropout(0.3)
     ↓
Dense(128, ReLU) + BatchNorm + Dropout(0.3)
     ↓
Dense(64, ReLU)
     ↓
Output Layer → Sigmoid (Binary Classification)
```

**Loss Function:** Binary Cross-Entropy  
**Optimizer:** Adam with learning rate scheduling  
**Regularization:** Dropout + L2

---

## 📊 Results

| Metric     | Score  |
|------------|--------|
| Accuracy   | ~85%   |
| Precision  | ~83%   |
| Recall     | ~87%   |
| F1-Score   | ~85%   |
| ROC-AUC    | ~0.91  |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

### Train the Model
```bash
python src/train.py
```

---

## 📦 Requirements

```
tensorflow>=2.10
keras
scikit-learn
pandas
numpy
streamlit
matplotlib
seaborn
rdkit
```

---

## 📈 Pipeline Overview

```
Raw Molecular Data
       ↓
Data Cleaning & Null Handling
       ↓
Feature Engineering (Molecular Descriptors)
       ↓
Train/Validation/Test Split (70/15/15)
       ↓
DNN Training with Early Stopping
       ↓
Evaluation (F1, Precision, Recall, AUC)
       ↓
Streamlit Deployment
```

---

## 🔬 Dataset

The dataset contains molecular descriptor features extracted from chemical compounds, labeled as active (1) or inactive (0) against a biological target. Features include physicochemical properties such as molecular weight, logP, hydrogen bond donors/acceptors, and topological polar surface area.

---

## 👤 Author

**Dev Kapania**  
B.Tech CSE (Big Data) — UPES  
Deep Learning Research Intern @ IIT Roorkee  
📧 devkapania2003@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/dev-kapania)

---

## 📄 License

This project is licensed under the MIT License.
