# 💼 Employee Salary Prediction - Niraj's Capstone Project

[Live Project ➜](https://salaryprediction-6979.onrender.com) • [GitHub Repo ➜](https://github.com/nirajj12/SalaryPrediction) • [Dataset ➜](https://www.kaggle.com/datasets/mrsimple07/salary-prediction-data)

---

## 📌 Overview
This project is part of a 6-week internship under **Edunet Foundation** with guidance from **Dr. Nanthini Mohan** and **Channabasava Yadav**. The objective is to predict employee salaries based on professional and personal attributes using a trained ML model and a Streamlit-based web interface.

---

## 🚀 Features
- 🔍 **Real-time salary prediction** based on age, experience, education, job title, gender, and location.
- 🎯 **ML pipeline** built with XGBoost and CatBoost achieving **R² Score ~ 0.91**.
- 🖥️ **Modern Streamlit UI** with insights, validation, salary visualization, and CSV export.
- ☁️ **Deployed on Render** for public access.

---

## 🧩 Project Architecture

```
Data Ingestion → Data Transformation → Model Training
         ↓                  ↓                  ↓
      ─────────────────→ Prediction Pipeline
     ↓                          ↓                  ↓
UI Development       Deployment
```

---

## 🛠️ How to Run Locally
```bash
# Clone the repository
git clone https://github.com/nirajj12/SalaryPrediction.git

# Navigate to the project folder
cd SalaryPrediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📁 Key Files & Folders
| Path                       | Description                              |
|----------------------------|------------------------------------------|
| `app.py`                   | Streamlit frontend code                  |
| `src/`                     | Contains all pipeline code               |
| `artifacts/`               | Stores models and transformed data       |
| `requirements.txt`         | Python dependencies                      |
| `Procfile`, `setup.py`     | For Render deployment                    |
| `predict_pipeline.py`      | Uses pre-trained model to predict salary |

---

## 📊 Model Performance
| Metric       | Value            |
|--------------|------------------|
| R² Score     | 0.87             |
| RMSE         | ₹15,000–₹16,000  |
| MAE          | ~₹9,000          |
| Best Model   | Ridge / Lasso / Linear Regression |

---

## 💡 How It Works
1. **User Input:** Age, Gender, Education, Experience, Job Title, Location.
2. **Data Conversion:** `CustomData` class turns input into DataFrame.
3. **Transformation:** Preprocessed using `preprocessor.pkl`.
4. **Prediction:** Predicted using `model.pkl`.
5. **Display:** Result shown with insights and visualization.

---

## 🧠 Technologies Used
- Python 3.10+
- Streamlit
- scikit-learn, XGBoost, CatBoost
- pandas, numpy, seaborn, matplotlib
- plotly (charts), dill/pickle (model storage)
- Render (cloud deployment)
- Git & GitHub

---

## 🔮 Future Improvements
- 🔐 User login & authentication
- 📈 Real-time salary API integration
- 🌐 Multilingual support
- 🧾 Add industry/domain filters

---

## 🙌 Acknowledgments
- 📚 **Dataset**: [Kaggle](https://www.kaggle.com/datasets/mrsimple07/salary-prediction-data)
- 🎓 **Mentors**: Dr. Nanthini Mohan & Channabasava Yadav
- 🤝 **Internship**: Edunet Foundation & IBM SkillsBuild (via AICTE)

---

> Built with ❤️ by Niraj
