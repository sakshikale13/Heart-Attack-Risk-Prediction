 # The main Flask application file:

from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load('heart_attack_model.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data from user input
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Prepare features for prediction
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Make prediction using the trained model
        prediction = model.predict(features)

        # Output the prediction result
        if prediction[0] == 1:
            result = "⚠️ High risk of heart attack. Please consult a doctor."
        else:
            result = "✅ Low risk of heart attack."

        return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
