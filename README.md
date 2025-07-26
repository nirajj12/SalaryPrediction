**🌟 Employee Salary Prediction - Complete Pipeline Documentation**

📍 **Live Project Link**: [Salary Prediction App](https://salaryprediction-6979.onrender.com)  
🔗 **GitHub Repository**: [GitHub - Niraj's Salary Prediction](https://github.com/nirajj12/SalaryPrediction)  
📊 **Dataset Used**: [Kaggle - Salary Prediction Data](https://www.kaggle.com/datasets/mrsimple07/salary-prediction-data)

---

### 🧩 Overview
Employee Salary Prediction is an end-to-end Machine Learning system designed and deployed to accurately predict the salary of an employee based on multiple features such as education, experience, age, job title, location, and gender.

This project was part of a 6-week internship under Edunet Foundation in collaboration with IBM SkillsBuild and AICTE.

---

### 📌 Project Objective
To build a fully functional ML-powered web application that allows users to:

- Input their professional data  
- Get a predicted salary in real-time  
- View insights and comparisons  
- Download prediction reports  
- Interact via an intuitive UI

---

### 🚧 Full Project Pipeline

#### 📁 1. Data Ingestion
**Objective**: Load the raw CSV dataset and split it into training and testing sets.

**Steps**:
- Load dataset from local directory or cloud.
- Save raw data to `artifacts/data.csv`.
- Perform an 80-20 train-test split.
- Save as `train.csv` and `test.csv` inside `artifacts/`.

**Code Module**: `src/components/data_ingestion.py`

#### 🔄 2. Data Transformation
**Objective**: Prepare the dataset for training.

**Steps**:
- Handle missing values using `SimpleImputer`.
- Convert categorical variables using `OneHotEncoder`.
- Normalize numeric variables with `StandardScaler`.
- Combine pipelines using `ColumnTransformer`.
- Save the transformation object as `preprocessor.pkl`.

**Code Module**: `src/components/data_transformation.py`

#### 🧠 3. Model Training
**Objective**: Train multiple regression models and select the best one using metrics.

**Models Used**:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- Ridge Regression
- Lasso Regression

**Tuning**: GridSearchCV used for hyperparameter tuning.  
**Metric**: Best model selected based on R² Score.  
**Final Model**: Saved as `model.pkl`.

**Code Module**: `src/components/model_trainer.py`

#### 📦 4. Prediction Pipeline
**Objective**: Use the saved model and transformer to make predictions from user input.

**Steps**:
- Accept input via a `CustomData` class.
- Transform the input using `preprocessor.pkl`.
- Predict using `model.pkl`.
- Return and display the result.

**Code Module**: `src/Pipeline/predict_pipeline.py`

#### 🎨 5. UI Development
**Framework Used**: Streamlit

**Features**:
- User inputs: Age, Gender, Education, Experience, Job Title, Location
- Sidebar with currency options and toggles
- Instant prediction with visual feedback and loading spinner
- Animated and styled UI using custom CSS
- Visualizations using Plotly charts
- Downloadable CSV report

**Code File**: `app.py`

#### 🌐 6. Deployment
**Platform**: Render.com

**Deployment Files**:
- `Procfile` (to define web service)
- `requirements.txt` (library dependencies)
- `setup.py` (project packaging)

**Live URL**: [https://salaryprediction-6979.onrender.com](https://salaryprediction-6979.onrender.com)

---

### 📊 Model Performance Summary
| **Metric**                  | **Value**              |
|----------------------------|------------------------|
| **R² Score (Best Model)**  | 0.87                   |
| **Root Mean Squared Error**| ₹15,616.67             |
| **Mean Absolute Error (MAE)** | ₹9,000 (approx.)    |
| **Best Model**             | Ridge / Lasso / Linear Regression |

---

### 💡 Technologies Used
| **Category**      | **Tools Used**                              |
|-------------------|----------------------------------------------|
| Language          | Python 3.10+                                |
| ML Libraries      | scikit-learn, XGBoost, CatBoost, AdaBoost   |
| Data Handling     | pandas, numpy                               |
| Preprocessing     | OneHotEncoder, StandardScaler, Imputers     |
| Visualization     | Matplotlib, Seaborn, Plotly                 |
| Web UI            | Streamlit                                   |
| Deployment        | Render                                      |
| Serialization     | pickle, dill                                |
| Version Control   | Git, GitHub                                 |

---

### 🧠 How It Works - User Flow

#### 👤 User Input:
- Age, Gender, Experience, Education, Job Title, Location

#### 🔁 Prediction Pipeline:
- Input → DataFrame → Preprocessing → ML Model → Output

#### 📊 Result Display:
- Predicted salary (in INR or selected currency)
- Career insights
- Visual salary comparison chart
- Downloadable prediction report

---

### 🎯 Future Enhancements
- 🔐 Add login/signup feature
- 🌍 Add location-based cost-of-living adjustments
- 📈 Show historical salary growth trends
- 🗣️ Add multi-language UI support
- 🧾 Add model explainability (SHAP, LIME)

---

### 🙌 Acknowledgments
- 📚 Dataset: [Kaggle - Salary Prediction Data](https://www.kaggle.com/datasets/mrsimple07/salary-prediction-data)
- 🤝 Internship Host: Edunet Foundation
- 🎓 Supported by: IBM SkillsBuild & AICTE
- 👨‍🏫 Mentors: Dr. Nanthini Mohan and Channabasava Yadav
