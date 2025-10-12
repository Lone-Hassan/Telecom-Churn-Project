import requests
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.modeling import load_data


# URL of your Flask API
url = "http://127.0.0.1:5000/predict_churn"

# Sample input data (replace with actual feature names and values)
df,X,y = load_data('../data/final/telecom_churn_features.csv')

sample_input = X.tail(2).to_dict(orient="records")
#print("Sample input data:", sample_input)

# Send POST request
response = requests.post(url, json=sample_input)

# Print response
if response.status_code == 200:
    print("Prediction result:")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error {response.status_code}:")
    print(response.text)