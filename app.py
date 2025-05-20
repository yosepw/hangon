# app.py (conceptual)
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model when the app starts
model = joblib.load('iris_logistic_regression_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Assuming data is a dictionary like {'sepal_length': 5.1, 'sepal_width': 3.5, ...}
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return jsonify({'prediction': int(prediction[0])}) # Convert numpy int to Python int

if __name__ == '__main__':
    app.run(debug=True)