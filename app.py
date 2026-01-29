from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# --- LOAD ASSETS ---
# CRITICAL CHANGE: Load 'best_model.pkl' to match the new 4algos.py logic
# This ensures the app always uses the winning model, whether it's RF, SVM, etc.
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and Scaler loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'best_model.pkl' or 'scaler.pkl' not found. Run 4algos.py first.")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Safety check: Ensure model is loaded
    if not model or not scaler:
        return render_template('index.html', result="System Error: Model not loaded. Please contact admin.")

    try:
        form_data = request.form
        
        # 1. Extract Features
        # We wrap this in try-except to catch empty or non-numeric inputs
        features = [
            int(form_data['trans_hour']),
            int(form_data['trans_day']),
            int(form_data['trans_month']),
            float(form_data['trans_amount']),
            int(form_data['age'])
        ]
        
        # 2. Preprocess
        features_array = [features]
        final_input = scaler.transform(features_array)
        
        # 3. Prediction (Probability Mode)
        # We get the probability of it being Fraud (Class 1)
        prob_fraud = model.predict_proba(final_input)[0][1]
        
        # --- TUNING THE SENSITIVITY ---
        # Any probability > 50% is considered suspicious.
        THRESHOLD = 0.50 

        risk_percentage = round(prob_fraud * 100, 2)
        
        # Prepare the result and CSS class for styling
        if prob_fraud >= THRESHOLD:
            result_text = f"🚨 FRAUD DETECTED (Risk: {risk_percentage}%)"
            css_class = "danger" # Use this class in HTML for Red color
        else:
            result_text = f"✅ Safe Transaction (Risk: {risk_percentage}%)"
            css_class = "success" # Use this class in HTML for Green color

        return render_template('index.html', result=result_text, css_class=css_class)

    except ValueError:
        # This catches errors if the user leaves a field blank or types text instead of numbers
        return render_template('index.html', result="Input Error: Please enter valid numbers for all fields.")
    except Exception as e:
        return render_template('index.html', result=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True,port = 5001)