# MentoPred - Mental Health Treatment Predictor

A machine learning project to predict whether an individual has sought treatment for a mental health condition based on survey data.

## Project Structure

```
app.py                  # Streamlit web application
src/                    # Source code directory
  ├── app.py            # Main application pipeline
  ├── data/             # Data processing modules
  │   └── preprocess.py # Data preprocessing functionality
  ├── models/           # Model-related modules
  │   ├── train.py      # Model training functionality
  │   └── predict.py    # Prediction functionality
  └── utils/            # Utility functions
      └── helpers.py    # Helper functions for logging, error handling
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages: pandas, numpy, scikit-learn, catboost, streamlit

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

To run the complete pipeline (preprocessing, feature engineering, and model training):

```python
from src.app import run_pipeline

# Run with default paths
results = run_pipeline()

# Or specify custom paths
results = run_pipeline(
    raw_data_path='data/raw/my_data.csv',
    processed_data_path='data/processed/my_processed_data.csv',
    engineered_data_path='data/processed/my_engineered_data.csv',
    config_path='configs/my_config.yml',
    model_path='models/my_model.pkl',
    report_path='artifacts/my_report.txt'
)
```

### Making Predictions

```python
from src.models.predict import MentalHealthPredictor

# Initialize predictor with model path
predictor = MentalHealthPredictor('models/final_model.pkl')

# Make a single prediction
input_data = {...}  # Dictionary with feature values
result = predictor.predict(input_data)

# Batch prediction
import pandas as pd
input_df = pd.read_csv('data/test_data.csv')
results = predictor.predict(input_df)
```

### Web Application

To run the Streamlit web application:

```bash
streamlit run app.py
```

## Model Details

- Algorithm: CatBoost Classifier
- Features: Survey responses related to workplace environment, mental health attitudes, and demographic information
- Target: Whether the individual has sought treatment for a mental health condition

## Configuration

Model parameters are defined in `configs/Catboost_final_param.yml`.