import os
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Loading model and vectorizer
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('dv.pkl', 'rb') as f:
    dv = pickle.load(f)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]
    
    return jsonify({
        "prediction": int(prediction),
        "probability_of_default": float(probability),
        "status": "APPROVED" if prediction == 0 else "DEFAULT"
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
