import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

# Configuration
PROCESSED_DIR = 'data/processed'
MODEL_DIR = 'src/models'

def train_model():
    print("Loading data...")
    train_path = os.path.join(PROCESSED_DIR, 'train.csv')
    test_path = os.path.join(PROCESSED_DIR, 'test.csv')
    
    if not os.path.exists(train_path):
        print("Error: Processed data not found. Run etl.py first.")
        return

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Separate Features and Target
    target = 'hospital_expire_flag'
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]
    
    print(f"Training features: {X_train.columns.tolist()}")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler for API
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("Scaler saved.")
    
    # Build Scikit-Learn Model
    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    
    # Save Model
    model_path = os.path.join(MODEL_DIR, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
