import os
import pickle
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator
from ..utils.helpers import CustomException


def load_model(model_path):
    """
    Load the trained model from file
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        raise CustomException(f"Error loading model from {model_path}: {e}")


class MentalHealthPredictor:
    """
    Class for making predictions using the trained model
    """
    def __init__(self, model_path):
        """
        Initialize the predictor with the model path
        """
        self.model = load_model(model_path)
        
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = self.model.feature_names_in_.tolist()
        else:
            self.feature_names = None
            
        logging.info("MentalHealthPredictor initialized")
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction
        """
        try:
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            elif isinstance(input_data, pd.DataFrame):
                input_df = input_data.copy()
            else:
                raise CustomException("Input data must be a dictionary or pandas DataFrame")
            
            # Check for pipeline vs direct model
            if isinstance(self.model, BaseEstimator) and not hasattr(self.model, 'steps'):
                # Direct model - ensure features are in the right format
                if self.feature_names and set(input_df.columns) != set(self.feature_names):
                    raise CustomException(f"Input features don't match model features. Expected: {self.feature_names}")
            
            return input_df
            
        except Exception as e:
            raise CustomException(f"Error preprocessing input data: {e}")
    
    def predict(self, input_data):
        """
        Make prediction using the trained model
        """
        try:
            input_df = self.preprocess_input(input_data)
            
            # Make predictions
            y_proba = self.model.predict_proba(input_df)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            
            # Create result dictionary
            results = {
                'prediction': y_pred.tolist(),
                'probability': y_proba.tolist(),
                'prediction_label': ['Yes' if p == 1 else 'No' for p in y_pred]
            }
            
            return results
            
        except Exception as e:
            raise CustomException(f"Error making prediction: {e}")
    
    def explain_prediction(self, input_data):
        """
        Explain prediction by extracting feature importances
        """
        try:
            # Check if the model has feature importances
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                
                if self.feature_names:
                    importance_dict = dict(zip(self.feature_names, importances))
                    sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    return sorted_importances[:10]  # Return top 10 features
                else:
                    return list(enumerate(importances))
            else:
                return "Model doesn't support feature importance extraction"
                
        except Exception as e:
            raise CustomException(f"Error explaining prediction: {e}")


def predict_single(model_path, input_data):
    """
    Make a single prediction using the trained model
    """
    try:
        predictor = MentalHealthPredictor(model_path)
        result = predictor.predict(input_data)
        return result
    except Exception as e:
        raise CustomException(f"Error in predict_single: {e}")


def batch_predict(model_path, input_file, output_file):
    """
    Make batch predictions using the trained model
    """
    try:
        logging.info(f"Loading data from {input_file}")
        input_data = pd.read_csv(input_file)
        
        logging.info("Initializing predictor")
        predictor = MentalHealthPredictor(model_path)
        
        logging.info("Making predictions")
        results = predictor.predict(input_data)
        
        # Add predictions to input data
        input_data['prediction'] = results['prediction']
        input_data['probability'] = results['probability']
        input_data['prediction_label'] = results['prediction_label']
        
        # Save predictions to file
        input_data.to_csv(output_file, index=False)
        logging.info(f"Predictions saved to {output_file}")
        
        return results
    except Exception as e:
        raise CustomException(f"Error in batch_predict: {e}")
