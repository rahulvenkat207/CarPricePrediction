from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model and data files
model_path = 'C:/krishnaikDataScienceMaterial/courPro/LinearRegression/project1/LinearRegressionModel.pkl'
data_path = 'C:/krishnaikDataScienceMaterial/courPro/LinearRegression/project1/Cleaned_Car_data.csv'

try:
    model = pickle.load(open(model_path, 'rb'))
    car = pd.read_csv(data_path)
except FileNotFoundError as e:
    print(f"File not found: {e}")
    model, car = None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Validate inputs
    if not all([company, car_model, year, fuel_type, driven]):
        return jsonify({"error": "Please provide all input fields"}), 400
    
    try:
        # Convert `driven` to integer
        driven = int(driven)

        # Predict price
        prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                                data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))

        return jsonify({"prediction": f"₹{np.round(prediction[0], 2)}"})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == '__main__':
    app.run(debug=True)
