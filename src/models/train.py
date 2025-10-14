import os
import pandas as pd
import numpy as np
import yaml
import pickle
import logging
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from ..utils.helpers import CustomException


def load_config(config_path):
    """
    Load configuration from YAML file
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise CustomException(f"Error loading config from {config_path}: {e}")


def load_data(file_path):
    """
    Load data from CSV file
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(f"Error loading data from {file_path}: {e}")


def save_model(model, file_path):
    """
    Save model to file
    """
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {file_path}")
    except Exception as e:
        raise CustomException(f"Error saving model to {file_path}: {e}")


def save_report(report, file_path):
    """
    Save classification report to file
    """
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(report)
        logging.info(f"Report saved to {file_path}")
    except Exception as e:
        raise CustomException(f"Error saving report to {file_path}: {e}")


def train_model(data_path, config_path, model_path, report_path=None):
    """
    Main function to train the model
    """
    try:
        logging.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)
        
        logging.info(f"Loading data from {data_path}")
        data = load_data(data_path)
        
        # Split data into features and target
        if 'treatment' in data.columns:
            X = data.drop('treatment', axis=1)
            y = data['treatment'].map({'Yes': 1, 'No': 0})
        else:
            raise CustomException("Target variable 'treatment' not found in the dataset")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logging.info("Initializing CatBoost model with parameters")
        catboost_params = config.get('catboost', {})
        model = CatBoostClassifier(
            iterations=catboost_params.get('iterations', 782),
            depth=catboost_params.get('depth', 3),
            learning_rate=catboost_params.get('learning_rate', 0.034),
            l2_leaf_reg=catboost_params.get('l2_leaf_reg', 0.257),
            border_count=catboost_params.get('border_count', 173),
            bagging_temperature=catboost_params.get('bagging_temperature', 0.863),
            random_strength=catboost_params.get('random_strength', 0.831),
            random_state=42,
            verbose=0
        )
        
        logging.info("Training CatBoost model")
        model.fit(X_train, y_train)
        
        # Evaluate model
        logging.info("Evaluating model")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate performance metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        class_report = classification_report(y_test, y_pred)
        
        logging.info(f"Model ROC AUC: {roc_auc:.4f}")
        logging.info(f"Classification Report:\n{class_report}")
        
        # Create a pipeline with the model
        pipeline = model
        
        # Save model
        logging.info(f"Saving model to {model_path}")
        save_model(pipeline, model_path)
        
        # Save classification report if specified
        if report_path:
            logging.info(f"Saving classification report to {report_path}")
            save_report(class_report, report_path)
        
        return model, roc_auc, class_report
        
    except Exception as e:
        raise CustomException(f"Error in training model: {e}")


if __name__ == "__main__":
    from ..utils.helpers import setup_logging
    
    setup_logging()
    
    data_path = os.path.join('data', 'processed', 'mental_health_engineered.csv')
    config_path = os.path.join('configs', 'Catboost_final_param.yml')
    model_path = os.path.join('models', 'final_model.pkl')
    report_path = os.path.join('artifacts', 'classification_report_train.txt')
    
    train_model(data_path, config_path, model_path, report_path)
