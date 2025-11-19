import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
MODEL_FILE = 'xgb_model.pkl'
DV_FILE = 'dv.pkl'

# XGBoost parameters 
params = {
    'n_estimators': 50,
    'max_depth': 3,
    'learning_rate': 0.2,
    'subsample': 0.6,
    'colsample_bytree': 0.9,
    'min_child_weight': 1,
    'gamma': 0,
    'random_state': SEED,
    'eval_metric': 'logloss'
}
\

df = pd.read_csv('loan_data.csv')

# Split into train/val/test
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['loan_status'])
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=SEED, stratify=df_full_train['loan_status'])

# Separate features from target
y_train = df_train['loan_status'].values
y_val = df_val['loan_status'].values
y_test = df_test['loan_status'].values

train_dict = df_train.drop('loan_status', axis=1).to_dict(orient='records')
val_dict = df_val.drop('loan_status', axis=1).to_dict(orient='records')
test_dict = df_test.drop('loan_status', axis=1).to_dict(orient='records')

# One-hot encode the features
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)
X_test = dv.transform(test_dict)

# Training the model
print('Training model...')
model = XGBClassifier(**params)
model.fit(X_train, y_train)

# Checking performance on validation and test sets
val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f'Validation AUC: {val_auc:.4f}')
print(f'Test AUC: {test_auc:.4f}')

# Save everything for later
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
    
with open(DV_FILE, 'wb') as f:
    pickle.dump(dv, f)

print(f'Model saved to {MODEL_FILE}')
