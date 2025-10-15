# MentoPred: Mental Health Treatment Prediction in the Tech Industry

MentoPred is a machine learning application that predicts whether an individual in the tech industry has sought treatment for mental health conditions based on their survey responses related to workplace factors and personal attitudes.

## Overview

In tech workplaces, mental health is increasingly recognized as crucial yet stigmatized. The goal of this project is to build a model that predicts whether an individual has sought treatment for a mental health condition based on their responses to survey questions about demographics, workplace culture, attitudes, and support systems. The model aims to be accurate, interpretable, and useful for informing policy or workplace intervention.

## Project Structure

```
.
├── app.py                 # Streamlit application for prediction
├── configs/               # Model configuration files 
│   ├── baseline.yml                   # Baseline model configuration
│   └── Catboost_final_param.yml       # Optimized model parameters
├── data/                  # Data files
│   ├── processed/         # Processed data files
│   │   ├── mental_health_cleaned.csv   # Cleaned dataset
│   │   ├── mental_health_encoded.csv   # Encoded dataset
│   │   ├── mental_health_engineered.csv # Dataset with engineered features
│   │   └── unique_values.csv           # Unique categorical values
│   └── raw/               # Original data files
│       └── mental_health_survey.csv    # Raw survey data
├── models/                # Trained models
│   ├── encoder_final.pkl.dvc          # DVC-tracked encoder for preprocessing
│   ├── encoding_pipeline.pkl          # Sklearn pipeline for encoding
│   ├── final_model.pkl               # Actual model file
│   └── final_model.pkl.dvc           # DVC-tracked optimized CatBoost model
├── Notebooks/             # Jupyter notebooks for analysis
│   ├── data_preparation.ipynb         # Data cleaning and feature engineering
│   ├── exploratory_data_analysis.ipynb # EDA notebook
│   └── modeling.ipynb                  # Model development and evaluation
├── pyproject.toml         # Poetry project configuration
├── requirements.txt       # Package dependencies
└── src/                   # Source code modules
    ├── app.py             # Application logic
    ├── data/              # Data processing modules
    ├── models/            # Model training and prediction
    ├── preprocessing/     # Data preprocessing utilities
    ├── utils/             # Helper functions
    └── __init__.py        # Package initialization
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/utkarsh820/mentopred.git
   cd mentopred
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the environment:
   ```
   # Optional: Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

## Running the Application

To run the Streamlit app, use the following command:

```
streamlit run app.py
```

This will start the application and open it in your default web browser.

## Using the Application

1. Fill out the survey form with information about:
   - Personal demographics (age, gender, country)
   - Work environment (company size, remote work, tech company)
   - Mental health support (benefits, wellness programs, care options)
   - Workplace attitudes (coworker/supervisor support, leave policy)
   - Privacy concerns (anonymity, consequences of disclosure)

2. Click "Predict" to see the results.

3. The app will display:
   - The prediction of whether the individual has sought mental health treatment
   - The probability score for the prediction
   - Key factors contributing to the prediction
   - Suggestions for workplace mental health support measures

## Technical Approach

### Data Processing
- **Data Cleaning**: Standardized column names, handled missing values, and normalized categorical variables
- **Feature Engineering**: Created composite features capturing workplace support, mental health awareness, and privacy concerns
- **Encoding Strategy**:
  - Binary Variables: Label Encoding (0, 1)
  - Ordinal Variables: Custom ordinal encoding preserving meaningful order
  - Nominal Variables: One-Hot Encoding for low cardinality categories
  - Country Variable: Target encoding with smoothing for this high-cardinality feature
  - Numerical Variables: Standard scaling

### Modeling Process
- **Initial Screening**: Evaluated multiple algorithms including LogisticRegression, RandomForest, XGBoost, LightGBM, and CatBoost
- **Feature Selection**: Analyzed feature importance to understand key predictors
- **Hyperparameter Optimization**: Used Bayesian optimization with Optuna to fine-tune the CatBoost model
- **Model Evaluation**: Employed 5-fold cross-validation with ROC-AUC scoring
- **Threshold Optimization**: Fine-tuned the prediction threshold to balance precision and recall

## Model Performance

- **ROC-AUC Score**: 0.88-0.91 (Test/CV)
- **Accuracy**: ~80-83% (with optimized threshold)
- **Top Predictive Features**:
  1. Family history of mental health issues
  2. Work interference with mental health
  3. Privacy concern score
  4. Workplace support score
  5. Mental health awareness in the workplace

## Data Privacy

All data entered in the application is processed locally and is not stored or shared with any third parties. The prediction is made entirely on the user's device.

## For Developers

### Model Retraining

If you want to retrain the model or modify the application:

1. Explore the notebooks in the `Notebooks/` directory:
   - `data_preparation.ipynb`: Data cleaning and feature engineering
   - `exploratory_data_analysis.ipynb`: Data visualization and insights
   - `modeling.ipynb`: Model development and evaluation

2. Experiment with your own feature engineering:
   ```python
   # Example of creating a new composite feature
   df['new_feature'] = df['feature1'] * df['feature2']
   ```

3. Track your experiments using MLflow:
   ```python
   with mlflow.start_run(run_name="experiment_name"):
       mlflow.log_param("param_name", param_value)
       mlflow.log_metric("metric_name", metric_value)
   ```

## Experiments Tracking

View experiment results and model performance metrics:
[DagsHub MLflow Experiments](https://dagshub.com/utkarsh820/mentopred.mlflow/#/experiments/0?viewStateShareKey=6cdb275f8e4303e9a247aa495293ca5b0c043c05459e19913e1d770d4837d59b&compareRunsMode=TABLE)

## License

This project is licensed under the terms of the license included in the repository.
