from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("random_forest_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index_Churn.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract inputs
    CreditScore = int(data['CreditScore'])
    Geography = data['Geography']
    Age = int(data['Age'])
    Tenure = int(data['Tenure'])
    Balance = float(data['Balance'])
    NumOfProducts = int(data['NumOfProducts'])
    HasCrCard = int(data['HasCrCard'])
    IsActiveMember = int(data['IsActiveMember'])
    EstimatedSalary = float(data['EstimatedSalary'])
    Gender = data['Gender']

    # One-hot encoding for Geography
    Geography_Germany = 1 if Geography == 'Germany' else 0
    Geography_Spain = 1 if Geography == 'Spain' else 0
    # France is implied if both above are 0

    # One-hot encoding for Gender
    Gender_Male = 1 if Gender == 'Male' else 0
    Gender_Female = 1 if Gender == 'Female' else 0

    # Prepare features in the correct order
    features = np.array([[CreditScore, Geography_Germany, Geography_Spain,
                          Age, Tenure, Balance, NumOfProducts,
                          HasCrCard, IsActiveMember, EstimatedSalary,
                          Gender_Female, Gender_Male]])

    # Predict
    prediction = model.predict(features)[0]
    result = "Client will EXIT" if prediction == 1 else "Client will STAY"
