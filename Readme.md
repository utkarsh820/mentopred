# MentoPred: Mental Health Treatment Prediction

MentoPred is a machine learning application that predicts whether an individual might benefit from mental health treatment based on workplace factors and personal attitudes.

## Overview

In tech workplaces, mental health is increasingly recognized as crucial yet stigmatized. The goal of this project is to build a model that predicts whether an individual has sought treatment for a mental health condition based on their responses to survey questions about demographics, workplace culture, attitudes, and support systems.

## Project Structure

```
.
├── app.py                 # Streamlit application for prediction
├── artifacts/             # Model evaluation artifacts
├── configs/               # Model configuration files 
├── data/                  # Data files
│   ├── processed/         # Processed data files
│   └── raw/               # Original data files
├── models/                # Trained models
│   └── final_model.pkl    # Combined encoder + model
├── Notebooks/             # Jupyter notebooks for analysis
└── src/                   # Source code modules
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

## Running the Application

To run the Streamlit app, use the following command:

```
streamlit run app.py
```

This will start the application and open it in your default web browser.

## Using the Application

1. Fill out the form with information about:
   - Personal demographics
   - Work environment
   - Mental health policies at work
   - Attitudes toward mental health
   - Workplace consequences

2. Click "Predict Treatment Need" to see the prediction results.

3. The app will display:
   - Whether treatment might be beneficial
   - The probability score for the prediction
   - A disclaimer about the non-clinical nature of the prediction

## Model Information

The machine learning model (`final_model.pkl`) combines both the encoder and the classifier. It was trained on the OSMI Mental Health in Tech Survey dataset and analyzes various factors to predict treatment needs.

## Data Privacy

All data entered in the application is processed locally and is not stored or shared with any third parties.

## Note for Developers

If you want to retrain the model or modify the application, refer to the Jupyter notebooks in the `Notebooks/` directory for the data preparation, exploratory analysis, and modeling process.

## License

This project is licensed under the terms of the license included in the repository.
