# 🏥 MIMIC-RiskLab: Clinical Mortality Prediction

MIMIC-RiskLab is a research-grade pipeline designed to process **Electronic Health Records (EHR)** to predict patient outcomes in Intensive Care Units. 


## 📖 Overview
Predicting mortality risk in the ICU is critical for resource allocation and clinical decision support. This project implements a deep learning approach to analyze patient vitals over time using the
**MIMIC-III** dataset.
[🚀 Launch Temporal Risk Dashboard](https://huggingface.co/spaces/abdalla6805/MIMIC-Risk-Predictor)

## 🛠 Features
- **Data Cleaning:** Automated handling of irregular time-series data and missing clinical vitals.
- **Feature Engineering:** Calculation of clinical scores (SAPS II, SOFA) and rolling statistics.
- **Deep Learning Architecture:** Utilizes Long Short-Term Memory (LSTM) networks to capture temporal dependencies in patient health.

## 📊 Project Pipeline
1. **Extraction:** Querying PostgreSQL tables (Admissions, Patients, LabEvents).
2. **Preprocessing:** Outlier detection and mean-imputation.
3. **Training:** Model optimization using Adam optimizer and Cross-Entropy loss.
4. **Evaluation:** Performance measured via AUC-ROC and PR-Curve.


## 🚦 Data Access Disclaimer
Access to the MIMIC-III dataset is restricted. You must complete the CITI training and request access via [PhysioNet](https://physionet.org/). **No PHI (Protected Health Information) is included in this repository.**

## 📈 Performance (Sample Results)
| Model | AUC-ROC | Accuracy |
| :--- | :--- | :--- |
| Random Forest | 0.78 | 81% |
| **LSTM (Current)** | **0.88** | **89%** |

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main evaluation: `python src/main.py`
