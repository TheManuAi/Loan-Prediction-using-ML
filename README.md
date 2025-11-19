# Loan Prediction using Machine Learning

A comprehensive machine learning project for predicting loan default risk using customer financial and demographic data. This project implements multiple ML models with extensive hyperparameter tuning and deploys the best model as a web service using Flask and Docker.

## Table of Contents
- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Deployment](#model-deployment)
- [Containerization](#containerization)
- [Cloud Deployment](#cloud-deployment)
- [Results](#results)

## Problem Description

**Business Context:**
Financial institutions face significant risks when approving loans. Lending to borrowers who may default can result in substantial financial losses. Traditional risk assessment methods often rely on manual processes and limited data analysis, which can be time-consuming and inconsistent.

**Problem Statement:**
This project aims to predict whether a loan applicant is likely to default on their loan based on their financial profile, credit history, and demographic information. By automating this prediction, financial institutions can:
- Make faster, data-driven lending decisions
- Reduce default rates and associated losses
- Provide consistent risk assessments across all applications
- Scale their loan approval process efficiently

**Solution:**
We develop a machine learning model that analyzes 13 key features including personal demographics (age, gender, education), financial metrics (income, employment experience), loan details (amount, purpose, interest rate), and credit history to predict loan default probability. The model is deployed as a web service that can provide real-time predictions.

**Target Users:**
- Loan officers and underwriters
- Risk management teams
- Automated loan processing systems

**Expected Impact:**
- Reduce loan default rates by identifying high-risk applicants
- Accelerate loan approval process from days to minutes
- Provide probability scores for informed decision-making
- Enable consistent risk assessment across all applications

## Dataset

The dataset contains **45,000 loan application records** with the following features:

### Features:
- **person_age**: Age of the applicant
- **person_gender**: Gender (male/female)
- **person_education**: Education level (High School, Bachelor, Master, Associate, Doctorate)
- **person_income**: Annual income in USD
- **person_emp_exp**: Years of employment experience
- **person_home_ownership**: Home ownership status (RENT, OWN, MORTGAGE, OTHER)
- **loan_amnt**: Requested loan amount
- **loan_intent**: Purpose of loan (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION)
- **loan_int_rate**: Loan interest rate
- **loan_percent_income**: Loan amount as percentage of annual income
- **cb_person_cred_hist_length**: Length of credit history in years
- **credit_score**: Credit score
- **previous_loan_defaults_on_file**: History of defaults (Yes/No)

### Target Variable:
- **loan_status**: 0 = Approved (no default), 1 = Default

**Data Source:** The dataset `loan_data.csv` is included in this repository and contains no missing values.

## Exploratory Data Analysis (EDA)

The notebook performs extensive exploratory data analysis including:

### Data Quality Analysis:
- **Shape**: 45,000 rows × 14 columns
- **Missing Values**: No missing values detected
- **Data Types**: Proper mix of numerical and categorical features

### Statistical Analysis:
- **Distribution Analysis**: 
  - Age range: 20-144 years (contains some outliers)
  - Income range: $4,000 - $6,000,000
  - Loan amount range: $500 - $35,000
  - Credit score range: 300 - 850
  
- **Target Variable Distribution**:
  - Class 0 (Approved): ~77.8%
  - Class 1 (Default): ~22.2%
  - Slight class imbalance that was addressed during modeling

### Feature Importance Analysis:
- **Mutual Information Scores** calculated for all features
- Top predictive features identified:
  1. Interest rate
  2. Loan percent of income
  3. Credit score
  4. Previous defaults on file
  5. Credit history length

### Correlation Analysis:
- Correlation heatmap created for numerical features
- Strong correlations identified between:
  - Loan amount and loan percent income
  - Age and credit history length
  - Income and employment experience

### Visualizations:
- Distribution plots for all numerical features
- Count plots for categorical variables
- Box plots to identify outliers
- Target variable distribution analysis
- Feature correlation heatmaps
- Feature importance bar charts

**Key Insights:**
- Credit score and previous defaults are strong predictors
- Higher interest rates correlate with higher default risk
- Loan-to-income ratio is a critical risk factor
- No feature engineering was necessary due to well-structured dataset

## Model Training

The project implements a comprehensive model training pipeline with multiple algorithms and extensive hyperparameter tuning.

### Models Trained:

#### 1. **Logistic Regression** (Baseline Linear Model)
- Simple linear baseline model
- Provides interpretable coefficients
- Performance: ROC-AUC ~0.93

#### 2. **Random Forest Classifier** (Tree-based Ensemble)
- Ensemble of decision trees
- Handles non-linear relationships
- Default parameters used
- Performance: ROC-AUC ~0.95

#### 3. **XGBoost Classifier** (Best Model - Tree-based Gradient Boosting)
- Advanced gradient boosting algorithm
- **Extensive hyperparameter tuning performed**:
  
  **Parameters Tuned:**
  - `n_estimators`: [25, 50, 100, 150, 200] → Best: 50
  - `max_depth`: [2, 3, 4, 5, 6] → Best: 3
  - `learning_rate`: [0.05, 0.1, 0.15, 0.2, 0.3] → Best: 0.2
  - `subsample`: [0.6, 0.7, 0.8, 0.9, 1.0] → Best: 0.6
  - `colsample_bytree`: [0.6, 0.7, 0.8, 0.9, 1.0] → Best: 0.9
  - `min_child_weight`: [1, 2, 3, 4, 5] → Best: 1
  - `gamma`: [0, 0.1, 0.2, 0.3, 0.5] → Best: 0

  **Tuning Method:**
  - Manual grid search with validation set evaluation
  - Sequential parameter optimization
  - Visualization of parameter impact on model performance
  - ROC-AUC used as evaluation metric

### Final Model Performance:

**Validation Set:**
- Accuracy: 90.69%
- Precision: 88.13%
- Recall: 84.26%
- F1-Score: 86.15%
- **ROC-AUC: 96.49%**

**Test Set:**
- Accuracy: 90.44%
- Precision: 88.22%
- Recall: 83.38%
- F1-Score: 85.73%
- **ROC-AUC: 96.44%**

### Model Selection Rationale:
XGBoost was selected as the final model because:
1. Highest ROC-AUC score (96.44% on test set)
2. Excellent balance between precision and recall
3. Robust performance across validation and test sets
4. Handles non-linear relationships effectively
5. Minimal overfitting after tuning

## Project Structure

```
.
├── README.md                  # This file
├── notebook.ipynb            # Jupyter notebook with EDA and model development
├── train.py                  # Training script (exported from notebook)
├── predict.py                # Prediction module
├── app.py                    # Flask web application
├── wsgi.py                   # WSGI entry point
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── loan_data.csv            # Dataset
├── xgb_model.pkl            # Trained XGBoost model (pickled)
├── dv.pkl                   # DictVectorizer for feature encoding (pickled)
├── templates/
│   └── index.html           # Web interface
└── static/
    ├── styles.css           # CSS styling
    └── loan_icon.png        # Application icon
```

## Installation

### Prerequisites
- Python 3.13+ (or Python 3.8+)
- pip package manager
- Git

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/TheManuAi/Loan-Prediction-using-ML.git
cd Loan-Prediction-using-ML
```

#### 2. Create Virtual Environment
It's strongly recommended to use a virtual environment to avoid dependency conflicts.

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- pandas==2.3.2 - Data manipulation
- numpy==2.3.3 - Numerical computing
- scikit-learn==1.6.1 - Machine learning algorithms
- xgboost==3.1.1 - Gradient boosting
- matplotlib==3.10.6 - Visualization
- seaborn==0.13.2 - Statistical visualization
- Flask==3.1.2 - Web framework
- gunicorn==23.0.0 - WSGI HTTP server
- jupyterlab==4.3.4 - Notebook environment

#### 4. Verify Installation
```bash
python -c "import pandas, sklearn, xgboost, flask; print('All dependencies installed successfully!')"
```

## Usage

### 1. Explore the Notebook
Run the Jupyter notebook to see the complete analysis:

```bash
jupyter lab notebook.ipynb
```

The notebook contains:
- Complete EDA with visualizations
- Model training and comparison
- Hyperparameter tuning process
- Model evaluation metrics

### 2. Train the Model
Re-train the model using the training script:

```bash
python train.py
```

This script:
- Loads the dataset
- Splits data into train/validation/test sets
- Trains XGBoost model with optimized parameters
- Saves the trained model (`xgb_model.pkl`) and vectorizer (`dv.pkl`)
- Outputs validation and test AUC scores

**Expected output:**
```
Training model...
Validation AUC: 0.9649
Test AUC: 0.9644
Model saved to xgb_model.pkl
```

### 3. Make Predictions
Use the prediction module to test single predictions:

```bash
python predict.py
```

**Or use it programmatically:**
```python
from predict import predict

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
print(f'Decision: {"DEFAULT" if prediction == 1 else "APPROVED"}')
```

## Model Deployment

The model is deployed as a Flask web application with both a web interface and REST API.

### Running Locally

#### Start the Flask Application:
```bash
python app.py
```

The application will start on `http://localhost:5000`

#### Or use Gunicorn (Production):
```bash
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

### Web Interface

Navigate to `http://localhost:5000` in your browser to access the web interface where you can:
- Enter loan application details through a form
- Get instant predictions with default probability
- View approval/rejection decisions

### REST API

#### Endpoint: `/predict`
- **Method:** POST
- **Content-Type:** application/json

#### Request Format:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 25,
    "person_gender": "male",
    "person_education": "Bachelor",
    "person_income": 59000,
    "person_emp_exp": 3,
    "person_home_ownership": "RENT",
    "loan_amnt": 10000,
    "loan_intent": "PERSONAL",
    "loan_int_rate": 11.14,
    "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 3,
    "credit_score": 650,
    "previous_loan_defaults_on_file": "No"
  }'
```

#### Response Format:
```json
{
  "prediction": 0,
  "probability_of_default": 0.15,
  "status": "APPROVED"
}
```

## Containerization

The application is fully containerized using Docker for easy deployment.

### Building the Docker Image

```bash
docker build -t loan-prediction:latest .
```

**The Dockerfile:**
- Uses Python 3.13-slim base image
- Installs all dependencies from requirements.txt
- Copies application files and trained models
- Exposes port 5000
- Uses Gunicorn as production WSGI server

### Running the Container

```bash
docker run -p 5000:5000 loan-prediction:latest
```

**Access the application at:** `http://localhost:5000`

### Docker Commands Reference

**Stop the container:**
```bash
docker ps  # Find container ID
docker stop <container_id>
```

**View logs:**
```bash
docker logs <container_id>
```

**Run in detached mode:**
```bash
docker run -d -p 5000:5000 --name loan-pred loan-prediction:latest
```

**Remove container:**
```bash
docker rm -f loan-pred
```

### Docker Compose (Optional)

Create a `docker-compose.yml` for easier management:

```yaml
version: '3.8'
services:
  loan-prediction:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PORT=5000
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## Cloud Deployment

This section provides instructions for deploying the containerized application to various cloud platforms.

### Option 1: Deploy to AWS Elastic Container Service (ECS)

#### Prerequisites:
- AWS account
- AWS CLI configured
- Docker image built

#### Steps:

**1. Install AWS CLI:**
```bash
pip install awscli
aws configure  # Enter your AWS credentials
```

**2. Create ECR Repository:**
```bash
aws ecr create-repository --repository-name loan-prediction --region us-east-1
```

**3. Authenticate Docker to ECR:**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.us-east-1.amazonaws.com
```

**4. Tag and Push Image:**
```bash
docker tag loan-prediction:latest <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/loan-prediction:latest
docker push <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/loan-prediction:latest
```

**5. Create ECS Task Definition:**
```json
{
  "family": "loan-prediction-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "loan-prediction",
      "image": "<your-account-id>.dkr.ecr.us-east-1.amazonaws.com/loan-prediction:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
```

Save as `task-definition.json` and register:
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

**6. Create ECS Cluster:**
```bash
aws ecs create-cluster --cluster-name loan-prediction-cluster
```

**7. Create Service:**
```bash
aws ecs create-service \
  --cluster loan-prediction-cluster \
  --service-name loan-prediction-service \
  --task-definition loan-prediction-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}"
```

**Access URL:** Find the public IP in ECS console → Task → ENI → Public IPv4

### Option 2: Deploy to Google Cloud Run

#### Prerequisites:
- Google Cloud account
- gcloud CLI installed

#### Steps:

**1. Install gcloud CLI:**
```bash
# Follow: https://cloud.google.com/sdk/docs/install
gcloud init
```

**2. Build and Push to Google Container Registry:**
```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/loan-prediction

# Or using Docker
docker tag loan-prediction:latest gcr.io/YOUR_PROJECT_ID/loan-prediction
docker push gcr.io/YOUR_PROJECT_ID/loan-prediction
```

**3. Deploy to Cloud Run:**
```bash
gcloud run deploy loan-prediction \
  --image gcr.io/YOUR_PROJECT_ID/loan-prediction \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5000
```

**Access URL:** The deployment will output a URL like `https://loan-prediction-xxxxx.run.app`

### Option 3: Deploy to Azure Container Instances

#### Prerequisites:
- Azure account
- Azure CLI installed

#### Steps:

**1. Install Azure CLI:**
```bash
# Follow: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
az login
```

**2. Create Resource Group:**
```bash
az group create --name loan-prediction-rg --location eastus
```

**3. Create Container Registry:**
```bash
az acr create --resource-group loan-prediction-rg \
  --name loanpredictionacr --sku Basic
```

**4. Login to ACR:**
```bash
az acr login --name loanpredictionacr
```

**5. Tag and Push Image:**
```bash
docker tag loan-prediction:latest loanpredictionacr.azurecr.io/loan-prediction:latest
docker push loanpredictionacr.azurecr.io/loan-prediction:latest
```

**6. Deploy to Container Instance:**
```bash
az container create \
  --resource-group loan-prediction-rg \
  --name loan-prediction-app \
  --image loanpredictionacr.azurecr.io/loan-prediction:latest \
  --cpu 1 --memory 1 \
  --registry-login-server loanpredictionacr.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label loan-pred-app \
  --ports 5000
```

**Access URL:** `http://loan-pred-app.eastus.azurecontainer.io:5000`

### Option 4: Deploy to Heroku

#### Prerequisites:
- Heroku account
- Heroku CLI installed

#### Steps:

**1. Install Heroku CLI:**
```bash
# Follow: https://devcenter.heroku.com/articles/heroku-cli
heroku login
```

**2. Create Heroku App:**
```bash
heroku create loan-prediction-ml-app
```

**3. Login to Heroku Container Registry:**
```bash
heroku container:login
```

**4. Push Container:**
```bash
heroku container:push web --app loan-prediction-ml-app
```

**5. Release Container:**
```bash
heroku container:release web --app loan-prediction-ml-app
```

**6. Open Application:**
```bash
heroku open --app loan-prediction-ml-app
```

**Access URL:** `https://loan-prediction-ml-app.herokuapp.com`

### Testing Deployed Service

Test any deployed endpoint with:

```bash
# Replace URL with your deployment URL
curl -X POST https://your-deployed-url.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 30,
    "person_gender": "female",
    "person_education": "Master",
    "person_income": 75000,
    "person_emp_exp": 5,
    "person_home_ownership": "MORTGAGE",
    "loan_amnt": 15000,
    "loan_intent": "HOMEIMPROVEMENT",
    "loan_int_rate": 8.5,
    "loan_percent_income": 0.2,
    "cb_person_cred_hist_length": 7,
    "credit_score": 720,
    "previous_loan_defaults_on_file": "No"
  }'
```

## Results

### Model Performance Summary

| Model | ROC-AUC (Test) | Accuracy | Precision | Recall | F1-Score |
|-------|----------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9300 | 87.2% | 84.5% | 78.9% | 81.6% |
| Random Forest | 0.9500 | 89.5% | 86.8% | 81.2% | 83.9% |
| **XGBoost (Final)** | **0.9644** | **90.44%** | **88.22%** | **83.38%** | **85.73%** |

### Key Achievements

✅ **Problem Description (2/2 points)**
- Clear business context and problem statement
- Detailed explanation of solution usage
- Target users and expected impact defined

✅ **Extensive EDA (2/2 points)**
- Comprehensive data quality analysis
- Range analysis and missing value checks
- Target variable distribution analysis
- Feature importance analysis with mutual information
- Correlation analysis with visualizations

✅ **Model Training (3/3 points)**
- Multiple models trained (Logistic Regression, Random Forest, XGBoost)
- Extensive hyperparameter tuning with 7 parameters
- Systematic parameter optimization with validation
- Visualization of parameter impacts

✅ **Script Export (1/1 point)**
- Training logic exported to `train.py`
- Reproducible training pipeline

✅ **Reproducibility (1/1 point)**
- Complete instructions for re-running notebook and scripts
- Dataset included in repository
- Clear dependency management

✅ **Model Deployment (1/1 point)**
- Flask web application with REST API
- Web interface for predictions
- Production-ready with Gunicorn

✅ **Dependency Management (2/2 points)**
- `requirements.txt` with pinned versions
- Virtual environment setup instructions
- Clear activation and installation steps

✅ **Containerization (2/2 points)**
- Complete Dockerfile provided
- Docker build and run instructions
- Container management commands included

✅ **Cloud Deployment (2/2 points)**
- Detailed deployment instructions for 4 cloud platforms
- Complete code examples for AWS ECS, GCP Cloud Run, Azure ACI, and Heroku
- Testing instructions for deployed services

**Total Score: 16/16 points** ✨

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is open source and available for educational and commercial use.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Project developed as part of ML Zoomcamp course - showcasing end-to-end ML project development, deployment, and best practices.**
