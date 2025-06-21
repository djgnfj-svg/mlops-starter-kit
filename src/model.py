"""Machine Learning Model Definition"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
from typing import Tuple, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModel:
    """Simple ML Model wrapper for Iris classification"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.target_names = ['setosa', 'versicolor', 'virginica']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the model"""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.target_names)
        
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "model_type": "RandomForestClassifier",
            "n_features": X.shape[1]
        }
    
    def predict(self, features: list) -> dict:
        """Make prediction"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        probabilities = self.model.predict_proba(features_array)[0]
        
        return {
            "prediction": self.target_names[prediction],
            "prediction_id": int(prediction),
            "probabilities": {
                name: float(prob) 
                for name, prob in zip(self.target_names, probabilities)
            },
            "features": features
        }
    
    def save_model(self, path: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")