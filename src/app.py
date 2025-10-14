import os
import sys
import logging
from .data.preprocess import preprocess_data
from .models.train import train_model
from .models.predict import MentalHealthPredictor
from .utils.helpers import setup_logging, CustomException


def run_pipeline(raw_data_path=None, processed_data_path=None, engineered_data_path=None,
                config_path=None, model_path=None, report_path=None):
    """
    Run the complete pipeline: preprocessing, training, and evaluation
    """
    try:
        # Set default paths if not provided
        if raw_data_path is None:
            raw_data_path = os.path.join('data', 'raw', 'mental_health_survey.csv')
        
        if processed_data_path is None:
            processed_data_path = os.path.join('data', 'processed', 'mental_health_encoded.csv')
            
        if engineered_data_path is None:
            engineered_data_path = os.path.join('data', 'processed', 'mental_health_engineered.csv')
            
        if config_path is None:
            config_path = os.path.join('configs', 'Catboost_final_param.yml')
            
        if model_path is None:
            model_path = os.path.join('models', 'final_model.pkl')
            
        if report_path is None:
            report_path = os.path.join('artifacts', 'classification_report_train.txt')
        
        # Step 1: Preprocess data
        logging.info("Step 1: Preprocessing data")
        preprocess_data(raw_data_path, processed_data_path, create_features=False)
        
        # Step 2: Create engineered features
        logging.info("Step 2: Creating engineered features")
        preprocess_data(processed_data_path, engineered_data_path, create_features=True)
        
        # Step 3: Train model
        logging.info("Step 3: Training model")
        model, roc_auc, class_report = train_model(
            engineered_data_path, config_path, model_path, report_path
        )
        
        # Step 4: Log results
        logging.info(f"Pipeline completed successfully. Model ROC AUC: {roc_auc:.4f}")
        logging.info(f"Model saved to {model_path}")
        logging.info(f"Classification report saved to {report_path}")
        
        return {
            'model': model,
            'roc_auc': roc_auc,
            'class_report': class_report,
            'model_path': model_path
        }
        
    except Exception as e:
        error_msg = f"Error in pipeline: {e}"
        logging.error(error_msg)
        raise CustomException(error_msg)


def load_predictor(model_path=None):
    """
    Load the trained predictor for making predictions
    """
    try:
        if model_path is None:
            model_path = os.path.join('models', 'final_model.pkl')
            
        predictor = MentalHealthPredictor(model_path)
        logging.info(f"Predictor loaded from {model_path}")
        return predictor
    except Exception as e:
        error_msg = f"Error loading predictor: {e}"
        logging.error(error_msg)
        raise CustomException(error_msg)


if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Run the complete pipeline
    try:
        results = run_pipeline()
        logging.info("Pipeline completed successfully")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)
