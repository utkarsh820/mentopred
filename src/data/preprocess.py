import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import os
import logging
from ..utils.helpers import CustomException


class BinaryFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, binary_cols=None):
        self.binary_cols = binary_cols
        self.mappings = {}
        
    def fit(self, X, y=None):
        for col in self.binary_cols:
            if col in X.columns:
                unique_vals = X[col].unique()
                if len(unique_vals) <= 2:
                    self.mappings[col] = {val: i for i, val in enumerate(unique_vals)}
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapping).fillna(0).astype(int)
        return X_copy


class OrdinalFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.work_interfere_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Unknown': 4}
        self.leave_mapping = {'Very difficult': 0, 'Somewhat difficult': 1, 'Don\'t know': 2, 'Somewhat easy': 3, 'Very easy': 4}
        self.no_employees_mapping = {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5}
        self.coworkers_mapping = {'No': 0, 'Some of them': 1, 'Yes': 2}
        self.supervisor_mapping = {'No': 0, 'Some of them': 1, 'Yes': 2}
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if 'work_interfere' in X_copy.columns:
            X_copy['work_interfere'] = X_copy['work_interfere'].map(self.work_interfere_mapping).fillna(4)
            
        if 'leave' in X_copy.columns:
            X_copy['leave'] = X_copy['leave'].map(self.leave_mapping).fillna(2)
            
        if 'no_employees' in X_copy.columns:
            X_copy['no_employees'] = X_copy['no_employees'].map(self.no_employees_mapping).fillna(0)
            
        if 'coworkers' in X_copy.columns:
            X_copy['coworkers'] = X_copy['coworkers'].map(self.coworkers_mapping).fillna(0)
            
        if 'supervisor' in X_copy.columns:
            X_copy['supervisor'] = X_copy['supervisor'].map(self.supervisor_mapping).fillna(0)
            
        return X_copy


def create_preprocessing_pipeline():
    """
    Creates a preprocessing pipeline for the mental health dataset
    """
    # Feature groups
    categorical_features = ['gender', 'country']
    ordinal_features = ['work_interfere', 'leave', 'coworkers', 'supervisor', 'no_employees']
    binary_features = ['family_history', 'tech_company', 'remote_work', 'self_employed']
    yes_no_features = [
        'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity',
        'mental_health_consequence', 'phys_health_consequence',
        'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
    ]
    numerical_features = ['age']
    
    # Create the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            
            ('categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]), categorical_features + yes_no_features)
        ],
        remainder='passthrough'
    )
    
    # Complete pipeline
    encoding_pipeline = Pipeline([
        ('binary_encoder', BinaryFeatureEncoder(binary_cols=binary_features)),
        ('ordinal_encoder', OrdinalFeatureEncoder()),
        ('preprocessor', preprocessor)
    ])
    
    return encoding_pipeline


def create_engineered_features(df):
    """
    Create engineered features for the dataset
    """
    df_eng = df.copy()
    
    # Create composite features
    df_eng['workplace_support_score'] = (
        df_eng.get('benefits_Yes', 0) + 
        df_eng.get('care_options_Yes', 0) + 
        df_eng.get('wellness_program_Yes', 0) + 
        (df_eng.get('leave', 0) / 3)
    )

    df_eng['mental_health_awareness'] = (
        df_eng.get('mental_health_interview_Yes', 0) + 
        df_eng.get('phys_health_interview_Yes', 0) + 
        df_eng.get('mental_vs_physical_Yes', 0)
    )

    df_eng['social_support_score'] = (
        df_eng.get('coworkers_Yes', 0) * 2 + 
        df_eng.get('coworkers_Some of them', 0) + 
        df_eng.get('supervisor_Yes', 0) * 2 + 
        df_eng.get('supervisor_Some of them', 0)
    )

    df_eng['privacy_concern_score'] = (
        (1 - df_eng.get('anonymity_Yes', 0)) + 
        df_eng.get('mental_health_consequence_Yes', 0) + 
        df_eng.get('phys_health_consequence_Yes', 0) +
        df_eng.get('obs_consequence', 0)
    )

    df_eng['work_impact_ratio'] = df_eng.get('work_interfere', 0) / (df_eng.get('no_employees', 0) + 1)
    df_eng['family_history_x_support'] = df_eng.get('family_history', 0) * df_eng.get('workplace_support_score', 0)
    df_eng['remote_tech_interaction'] = df_eng.get('remote_work', 0) * df_eng.get('tech_company', 0)
    
    return df_eng


def load_data(file_path):
    """
    Load data from CSV file
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(f"Error loading data from {file_path}: {e}")


def save_data(df, file_path):
    """
    Save data to CSV file
    """
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        raise CustomException(f"Error saving data to {file_path}: {e}")


def preprocess_data(input_file, output_file, create_features=True):
    """
    Main preprocessing function to process data
    """
    try:
        logging.info(f"Loading data from {input_file}")
        df = load_data(input_file)
        
        logging.info("Creating preprocessing pipeline")
        pipeline = create_preprocessing_pipeline()
        
        logging.info("Applying preprocessing pipeline")
        if 'treatment' in df.columns:
            X = df.drop('treatment', axis=1)
            y = df['treatment']
            X_transformed = pipeline.fit_transform(X)
            
            # Convert to DataFrame
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
            df_transformed['treatment'] = y
        else:
            X_transformed = pipeline.fit_transform(df)
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        
        if create_features:
            logging.info("Creating engineered features")
            df_transformed = create_engineered_features(df_transformed)
        
        logging.info(f"Saving processed data to {output_file}")
        save_data(df_transformed, output_file)
        
        # Save the preprocessing pipeline
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        os.makedirs(model_dir, exist_ok=True)
        pipeline_path = os.path.join(model_dir, 'preprocessing_pipeline.pkl')
        
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        logging.info(f"Preprocessing pipeline saved to {pipeline_path}")
        
        return df_transformed
        
    except Exception as e:
        raise CustomException(f"Error in preprocessing data: {e}")
