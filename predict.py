import pickle


with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('dv.pkl', 'rb') as f:
    dv = pickle.load(f)

def predict(customer):
    X = dv.transform([customer])
    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0, 1]
    return int(y_pred), float(y_proba)

# Example usage
if __name__ == '__main__':
    customer = {
        'person_age': 25,
        'person_gender': 'male',
        'person_education': 'Bachelor',
        'person_income': 59000,
        'person_emp_exp': 3,
        'person_home_ownership': 'RENT',
        'loan_amnt': 10000,
        'loan_intent': 'PERSONAL',
        'loan_int_rate': 11.14,
        'loan_percent_income': 0.17,
        'cb_person_cred_hist_length': 3,
        'credit_score': 650,
        'previous_loan_defaults_on_file': 'No'
    }
    
    prediction, probability = predict(customer)
    print(f'Default probability: {probability:.2%}')
    print(f'Prediction: {"DEFAULT" if prediction == 1 else "APPROVED"}')
