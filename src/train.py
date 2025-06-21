"""Model Training Script"""

import os
import json
import logging
from datetime import datetime
from sklearn.datasets import load_iris
import numpy as np
from model import MLModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model():
    """Train and save the model"""
    logger.info("Starting training pipeline...")
    
    # Load data
    logger.info("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Initialize and train model
    model = MLModel()
    metrics = model.train(X, y)
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{timestamp}.pkl"
    model_path = os.path.join("models", model_filename)
    
    model.save_model(model_path)
    
    # Save training metadata
    metadata = {
        "timestamp": timestamp,
        "model_path": model_path,
        "metrics": metrics,
        "dataset": "iris",
        "features": model.feature_names,
        "targets": model.target_names
    }
    
    metadata_path = os.path.join("models", f"metadata_{timestamp}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create symlink to latest model
    latest_link = os.path.join("models", "latest_model.pkl")
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(model_filename, latest_link)
    
    logger.info(f"Training completed. Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    
    return metadata


if __name__ == "__main__":
    train_model()