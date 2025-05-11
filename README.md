
# 📊 Predictive Model for Clinical Excellence in Cardiac Surgery

## 🚀 Overview
This project aims to develop a machine learning-powered decision support tool that predicts post-operative outcomes—such as mortality, renal failure, stroke, and ventilation length—for cardiac surgery patients. Our objective is to assist surgical teams in preoperative risk stratification and process optimization using real patient data.

## 🎯 Objectives
- **Predict operative mortality and major complications** using preoperative clinical indicators.
- **Support clinical excellence** by aligning predictions with STS star rating benchmarks.
- **Reduce human error** by automating predictions previously calculated manually via the STS tool.

## 🧠 Methodology

| Step | Description |
|------|-------------|
| 🧹 Preprocessing | Cleaned and scaled structured STS-like data with encoded clinical outcomes. |
| 📊 EDA | Generated pairplots and correlation heatmaps to uncover trends and predictive potential. |
| 🛠 Modeling | Used **Gradient Boosting Regressor** with hyperparameter tuning via `RandomizedSearchCV`. |
| 📈 Evaluation | Performance measured using **R² Score** and **MSE** with residual & learning curve plots. |

## 🔍 Sample Results

| Metric | Value |
|--------|-------|
| R² Score | 0.78 |
| Mean Squared Error | 0.04 |
| Top Features | Renal Failure, Prolonged Ventilation, Short Hospital Stay |

## 📁 Project Structure
```
📦 Cardiac-Surgery-Predictive-Model
│
├── main.py                            # Full modeling pipeline (preprocessing → training → insights)
├── Surgical_Predictive_Model_Final.ipynb  # Annotated Jupyter notebook for EDA + modeling
├── main_fresh_complete_data_only.ipynb    # Fresh clean data workflow
├── ModelData_2425.xlsx               # Input dataset (STS-style clinical records)
├── best_gradient_boosting_model.pkl # Saved final model for deployment
├── Project proposal.pdf              # Project scope and ML pipeline plan
├── OU_ELM Engineers.pdf              # Clinical sponsor context + domain requirements
└── README.md                         # [YOU ARE HERE]
```

## 🧪 Features & Techniques
- STS-aligned outcomes: mortality, AKI, stroke, ventilation, etc.
- Real-world clinical use case from HCA Healthcare
- Gradient Boosting with Random Search
- Residual & importance plots for explainability
- Exportable `.pkl` model for downstream integration

## 🛠️ Tech Stack
- **Language**: Python
- **Modeling**: Scikit-learn, XGBoost (optional), TensorFlow (future work)
- **EDA**: Seaborn, Matplotlib
- **Dev Tools**: VS Code, Jupyter, Git

## 🏥 Clinical Relevance
Based on the guidelines from the Society of Thoracic Surgeons (STS) and a custom Excel-based risk scoring system used by HCA Houston Clear Lake, this project automates and augments the process of predicting patient-specific risk factors—supporting the goal of achieving 3-star STS surgical ratings.

## 📌 Next Steps
- Expand to AVR, MVR, AVR/CABG, and TAVR predictive models
- Integrate web-based interface for clinical users
- Incorporate longitudinal data for more robust forecasting

## 🙋‍♂️ Author
**Abdulmalik Ajisegiri**  
Systems Engineering MS Student | Model Risk Intern @ DTCC  
📧 abdulmalik.ajisegiri@ou.edu
