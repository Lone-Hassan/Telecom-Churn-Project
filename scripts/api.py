from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)


# Load the model pipline
model = joblib.load("../models/final_churn_pipeline.pkl")
feature_names = model.named_steps["onehotencoder"].get_feature_names_out()

@app.route('/')
def home():
    return jsonify({"message": "Churn Prediction API is running!"})


@app.route('/predict_churn', methods=['POST'])
def predict_churn_endpoint():
    
    try:
        
        # Get data from request
        data = request.get_json()
        # Convert to DataFrame
        df_input = pd.DataFrame(data)
        # Make prediction
        risk_level = []
        probability = model.predict_proba(df_input)[:, 1]
        prediction = model.predict(df_input)
        print(probability)
        print(prediction)
        for prob in probability:
            # Determine risk level
            if prob > 0.7:
                risk_level.append('High')
            elif prob > 0.4:
                risk_level.append('Medium')
            else:
                risk_level.append('Low')
        response = {
            'churn_probability': [float(x) for x in probability],
            'churn_prediction': [int(x) for x in prediction],
            'risk_level': risk_level,
            'status': 'success'
            }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400
    
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)