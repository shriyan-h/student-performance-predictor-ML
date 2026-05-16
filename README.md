# 🎓 Student Performance Predictor

A machine learning web app that predicts a student's final grade 
based on their academic profile.

## 🔗 Live Demo
[Click here to try the app](<your-streamlit-link-after-deploy>)

## 📊 Model Performance
- **R2 Score:** 0.863
- **RMSE:** 1.15 / 20

## 🧠 How It Works
The model is trained on real student data from two Portuguese 
schools. It uses Linear Regression to predict the final grade (G3) 
based on 6 key features.

## 📁 Features Used
| Feature | Description |
|---|---|
| Age | Student's age |
| Study Time | Weekly study hours |
| Failures | Number of past failures |
| Absences | Number of school absences |
| G1 | First period grade |
| G2 | Second period grade |

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib
