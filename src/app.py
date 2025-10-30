from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import Dict
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection",
    version="1.0.0"
)

# Load model and scaler at startup
print("Loading model...")
with open('models/optimized/xgboost_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/optimized/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/optimized/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("✓ Model loaded successfully!")

# Define input data structure
class Transaction(BaseModel):
    """
    Transaction data structure
    """
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    class Config:
        schema_extra = {
            "example": {
                "Time": 406,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62
            }
        }

def engineer_features(transaction_data: Dict) -> pd.DataFrame:
    """
    Engineer features from raw transaction data
    """
    df = pd.DataFrame([transaction_data])

    # Time features
    df['Hour'] = (df['Time'] / 3600) % 24
    df['Day'] = (df['Time'] / 86400).astype(int)
    df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)

    # Amount features
    df['Amount_Log'] = np.log1p(df['Amount'])
    df['Is_Round_Amount'] = (df['Amount'] % 10 == 0).astype(int)

    # Statistical features from V columns
    v_columns = [col for col in df.columns if col.startswith('V')]
    df['V_sum'] = df[v_columns].sum(axis=1)
    df['V_mean'] = df[v_columns].mean(axis=1)
    df['V_std'] = df[v_columns].std(axis=1)
    df['V_max'] = df[v_columns].max(axis=1)
    df['V_min'] = df[v_columns].min(axis=1)

    # Select only the features used in training
    return df[feature_names]

@app.get("/")
def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict")
def predict_fraud(transaction: Transaction) -> Dict:
    """
    Predict if a transaction is fraudulent

    Returns:
        - prediction: 0 (legitimate) or 1 (fraud)
        - probability: confidence score (0-1)
        - risk_level: low, medium, high
        - recommendation: action to take
    """
    try:
        # Convert to dictionary
        transaction_dict = transaction.dict()

        # Engineer features
        features = engineer_features(transaction_dict)

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0, 1]

        # Determine risk level
        if probability < 0.3:
            risk_level = "low"
            recommendation = "Approve transaction"
        elif probability < 0.7:
            risk_level = "medium"
            recommendation = "Request additional verification (3D Secure, OTP)"
        else:
            risk_level = "high"
            recommendation = "Decline transaction and alert cardholder"

        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "transaction_amount": transaction.Amount,
            "timestamp": transaction.Time
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
def predict_fraud_batch(transactions: list[Transaction]) -> Dict:
    """
    Predict fraud for multiple transactions at once
    """
    try:
        results = []

        for transaction in transactions:
            transaction_dict = transaction.dict()
            features = engineer_features(transaction_dict)
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0, 1]

            results.append({
                "prediction": int(prediction),
                "fraud_probability": float(probability),
                "amount": transaction.Amount
            })

        # Summary statistics
        total_transactions = len(results)
        flagged_frauds = sum(1 for r in results if r['prediction'] == 1)
        total_flagged_amount = sum(r['amount'] for r in results if r['prediction'] == 1)

        return {
            "results": results,
            "summary": {
                "total_transactions": total_transactions,
                "flagged_frauds": flagged_frauds,
                "fraud_rate": flagged_frauds / total_transactions,
                "total_flagged_amount": total_flagged_amount
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)