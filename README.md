# Employee Salary Prediction System

A Machine Learning web application that predicts employee salary based on user inputs such as experience, education, performance, and other factors.

## Live Demo
🔗 https://employee-salary-predictor-s.streamlit.app/

## Features
- Predicts employee salary using Machine Learning
- Classifies salary level (Low / Medium / High)
- User-friendly Streamlit interface
- Fast and accurate predictions

## Algorithms Used
- Random Forest Regressor
- Logistic Regression Classifier

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Project Structure
- `app.py` – Main Streamlit application
- `requirements.txt` – Required libraries
- `model.pkl` – Trained ML model
- `dataset.csv` – Dataset used for training

## How to Run Locally

1. Clone repository

```bash
git clone <your-github-link>
cd employee-salary-prediction-system

2. Install dependencies
pip install -r requirements.txt

3. Run app
streamlit run app.py


## Deployment

Deployed using Streamlit Community Cloud.

## Output

The system predicts:

Exact salary amount
Salary level (Low / Medium / High)
