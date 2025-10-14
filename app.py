import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Define the custom encoder class to match what was used in training
class MentoPredEncoder(BaseEstimator, TransformerMixin):
    """Custom encoder class for MentoPred project"""
    def __init__(self):
        self.gender_mapping = {}
        self.work_interfere_mapping = {}
        self.no_employees_mapping = {}
        self.leave_mapping = {}
        self.binary_cols = []
        self.nominal_cols = []
        self.numerical_cols = []
        self.country_encoding = {}
        self.global_mean = 0.0
        self.scaler = None
        self.expected_columns = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # This will be implemented in our custom encoding function
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

# Set page configuration
st.set_page_config(
    page_title="MentoPred - Mental Health Treatment Predictor",
    page_icon="🧠",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join('models', 'final_model.pkl')
        
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        
        # Debug information about the pipeline
        if isinstance(pipeline, Pipeline):
            st.success(f"Successfully loaded pipeline with {len(pipeline.steps)} steps")
            
            # Show pipeline steps in debug mode
            if st.session_state.get('debug_mode', False):
                st.write("Pipeline steps:")
                for i, (name, step) in enumerate(pipeline.steps):
                    st.write(f"Step {i}: {name} - {type(step)}")
            
            # Extract the model from the pipeline
            # The pipeline structure is ('encoder', string_path), ('model', actual_model)
            if len(pipeline.steps) > 1:
                model = pipeline.steps[1][1]  # Get the actual model (second step)
                st.success("Model extracted successfully from pipeline")
                
                # Debug model info
                if st.session_state.get('debug_mode', False):
                    if hasattr(model, 'feature_names_in_'):
                        st.write("Model expected features:", model.feature_names_in_.tolist())
                    
                return model
            else:
                st.error("Pipeline doesn't have the expected structure")
                return None
        else:
            st.warning("Loaded object is not a Pipeline, using it directly as model")
            return pipeline
    except FileNotFoundError:
        st.error(f"Model file not found at path: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Define the options for form fields based on unique values
form_options = {
    'gender': ['male', 'female', 'other'],
    'self_employed': ['Yes', 'No'],
    'family_history': ['Yes', 'No'],
    'work_interfere': ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown'],
    'no_employees': ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'],
    'remote_work': ['Yes', 'No'],
    'tech_company': ['Yes', 'No'],
    'benefits': ['Yes', 'No', "Don't know"],
    'care_options': ['Yes', 'No', 'Not sure'],
    'wellness_program': ['Yes', 'No', "Don't know"],
    'seek_help': ['Yes', 'No', "Don't know"],
    'anonymity': ['Yes', 'No', "Don't know"],
    'leave': ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"],
    'mental_health_consequence': ['Yes', 'No', 'Maybe'],
    'phys_health_consequence': ['Yes', 'No', 'Maybe'],
    'coworkers': ['Yes', 'No', 'Some of them'],
    'supervisor': ['Yes', 'No', 'Some of them'],
    'mental_health_interview': ['Yes', 'No', 'Maybe'],
    'phys_health_interview': ['Yes', 'No', 'Maybe'],
    'mental_vs_physical': ['Yes', 'No', "Don't know"],
    'obs_consequence': ['Yes', 'No']
}

# Function to convert categorical values to the format expected by the model
def preprocess_data(input_data):
    # Convert work_interfere to numerical value
    work_interfere_map = {
        'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Unknown': 0
    }
    input_data['work_interfere'] = work_interfere_map.get(input_data['work_interfere'], 0)
    
    # Convert leave to numerical value
    leave_map = {
        "Don't know": 2, 'Very difficult': 0, 'Somewhat difficult': 1, 'Somewhat easy': 3, 'Very easy': 4
    }
    input_data['leave'] = leave_map.get(input_data['leave'], 0)
    
    # Convert no_employees to numerical value
    no_employees_map = {
        '1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5
    }
    input_data['no_employees'] = no_employees_map.get(input_data['no_employees'], 0)
    
    # Standardize age (z-score normalization based on training data)
    mean_age = 32.0
    std_age = 8.5
    input_data['age'] = (input_data['age'] - mean_age) / std_age
    
    # One-hot encode categorical variables
    # For binary variables, just ensure they're 0/1
    binary_vars = ['self_employed', 'family_history', 'remote_work', 'tech_company', 'obs_consequence']
    for var in binary_vars:
        if var in input_data:
            input_data[var] = int(input_data[var])
    
    return pd.DataFrame([input_data])

# Function to encode the input data directly without a separate encoder
def encode_data(input_df):
    try:
        # Create encoded features for categorical variables
        encoded_features = {}
        
        # Handle gender encoding
        gender = input_df['gender'].iloc[0]
        encoded_features['gender_male'] = 1 if gender == 'male' else 0
        encoded_features['gender_other'] = 1 if gender == 'other' else 0
        
        # Handle categorical variables with Yes/No/Other values
        cat_vars = {
            'benefits': ['No', 'Yes', "Don't know"],
            'care_options': ['No', 'Yes', 'Not sure'],
            'wellness_program': ['No', 'Yes', "Don't know"],
            'seek_help': ['No', 'Yes', "Don't know"],
            'anonymity': ['No', 'Yes', "Don't know"],
            'mental_health_consequence': ['No', 'Yes', 'Maybe'],
            'phys_health_consequence': ['No', 'Yes', 'Maybe'],
            'coworkers': ['No', 'Yes', 'Some of them'],
            'supervisor': ['No', 'Yes', 'Some of them'],
            'mental_health_interview': ['No', 'Yes', 'Maybe'],
            'phys_health_interview': ['No', 'Yes', 'Maybe'],
            'mental_vs_physical': ['No', 'Yes', "Don't know"]
        }
        
        # Create one-hot encoded columns based on categorical values
        for var, options in cat_vars.items():
            value = input_df[var].iloc[0]
            for option in options:
                if option in ['No', 'Yes'] or option in ['Maybe', 'Not sure', "Don't know", 'Some of them']:
                    col_name = f"{var}_{option}"
                    encoded_features[col_name] = 1 if value == option else 0
        
        # Copy numeric columns directly
        for col in ['age', 'self_employed', 'family_history', 'work_interfere', 'no_employees', 
                   'remote_work', 'tech_company', 'leave', 'obs_consequence']:
            encoded_features[col] = input_df[col].iloc[0]
            
        # Add country encoding (simplified - just using a default value)
        # In a real scenario, this would need proper handling based on available countries
        encoded_features['country_encoded'] = 0.0
        
        # Debug print of encoded features
        st.write("Encoded features (debug):", encoded_features) if st.session_state.get('debug_mode', False) else None
        
        return pd.DataFrame([encoded_features])
    except Exception as e:
        st.error(f"Error encoding data: {e}")
        # Show more detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
        return None

# Function to make prediction
def predict(input_df, model, debug=False):
    try:
        # Store debug mode in session state for access in other functions
        st.session_state['debug_mode'] = debug
        
        # Encode the data
        encoded_df = encode_data(input_df)
        
        if encoded_df is None:
            return None
        
        # Debug - show the encoded dataframe if debug mode is on
        if debug:
            st.subheader("Debug Information")
            st.write("Original Input:")
            st.write(input_df)
            st.write("Encoded Features:")
            st.write(encoded_df)
            st.write("Encoded DataFrame Shape:", encoded_df.shape)
            st.write("Encoded DataFrame Columns:", encoded_df.columns.tolist())
        
        try:
            # Make prediction
            prediction = model.predict(encoded_df)
            prediction_proba = model.predict_proba(encoded_df)[0][1]
            
            return {
                'prediction': int(prediction[0]),
                'probability': float(prediction_proba),
                'treatment_needed': 'Yes' if prediction[0] == 1 else 'No'
            }
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            if debug:
                st.write("Model type:", type(model))
                st.write("Expected input features (if available):", getattr(model, 'feature_names_in_', "Not available"))
            raise e
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        # Show more detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
        return None

# Main app
def main():
    # Initialize session state for debug mode if it doesn't exist
    if 'debug_mode' not in st.session_state:
        st.session_state['debug_mode'] = False
    
    # Add a debug mode toggle in the sidebar
    with st.sidebar:
        st.title("Settings")
        debug_mode = st.checkbox("Debug Mode", value=st.session_state['debug_mode'])
        st.session_state['debug_mode'] = debug_mode
        
        if debug_mode:
            st.info("Debug mode enabled. You'll see additional technical information.")
    
    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🧠 MentoPred")
        st.markdown("<h3 style='text-align: center;'>Predicting Mental Health Treatment in the Tech Industry</h3>", unsafe_allow_html=True)
        st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Prediction Form", "About"])
    
    # Load the model (after setting debug_mode so it can use it)
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check if the file exists.")
        return
    
    with tab1:
        st.info("Fill out this form to predict whether mental health treatment might be beneficial based on workplace and personal factors.")
        
        # Create columns for form layout
        col1, col2 = st.columns(2)
        
        # Form for user input
        with st.form("prediction_form"):
            # Personal Information
            st.subheader("Personal Information")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", options=form_options['gender'])
            family_history = st.radio("Do you have a family history of mental illness?", options=form_options['family_history'])
            
            st.markdown("---")
            # Work Environment
            st.subheader("Work Environment")
            self_employed = st.radio("Are you self-employed?", options=form_options['self_employed'])
            no_employees = st.selectbox("How many employees does your company or organization have?", options=form_options['no_employees'])
            tech_company = st.radio("Is your employer primarily a tech company/organization?", options=form_options['tech_company'])
            remote_work = st.radio("Do you work remotely (outside of an office) at least 50% of the time?", options=form_options['remote_work'])
            
            st.markdown("---")
            # Mental Health at Work
            st.subheader("Mental Health at Work")
            col1, col2 = st.columns(2)
            
            with col1:
                work_interfere = st.selectbox("If you have a mental health condition, do you feel that it interferes with your work?", options=form_options['work_interfere'])
                benefits = st.selectbox("Does your employer provide mental health benefits?", options=form_options['benefits'])
                care_options = st.selectbox("Do you know the options for mental health care your employer provides?", options=form_options['care_options'])
            
            with col2:
                wellness_program = st.selectbox("Has your employer ever discussed mental health as part of an employee wellness program?", options=form_options['wellness_program'])
                seek_help = st.selectbox("Does your employer provide resources to learn more about mental health issues and how to seek help?", options=form_options['seek_help'])
                anonymity = st.selectbox("Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?", options=form_options['anonymity'])
            
            leave = st.selectbox("How easy is it for you to take medical leave for a mental health condition?", options=form_options['leave'])
            
            st.markdown("---")
            # Mental Health Consequences
            st.subheader("Mental Health Consequences")
            col1, col2 = st.columns(2)
            
            with col1:
                mental_health_consequence = st.selectbox("Do you think that discussing a mental health issue with your employer would have negative consequences?", options=form_options['mental_health_consequence'])
                phys_health_consequence = st.selectbox("Do you think that discussing a physical health issue with your employer would have negative consequences?", options=form_options['phys_health_consequence'])
            
            with col2:
                coworkers = st.selectbox("Would you be willing to discuss a mental health issue with your coworkers?", options=form_options['coworkers'])
                supervisor = st.selectbox("Would you be willing to discuss a mental health issue with your direct supervisor(s)?", options=form_options['supervisor'])
            
            st.markdown("---")
            # Mental Health in Interview Process
            st.subheader("Mental Health in Interview Process")
            col1, col2 = st.columns(2)
            
            with col1:
                mental_health_interview = st.selectbox("Would you bring up a mental health issue with a potential employer in an interview?", options=form_options['mental_health_interview'])
                phys_health_interview = st.selectbox("Would you bring up a physical health issue with a potential employer in an interview?", options=form_options['phys_health_interview'])
            
            with col2:
                mental_vs_physical = st.selectbox("Do you feel that your employer takes mental health as seriously as physical health?", options=form_options['mental_vs_physical'])
                obs_consequence = st.selectbox("Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?", options=form_options['obs_consequence'])
            
            # Submit button
            submitted = st.form_submit_button("Predict Treatment Need")
            
            if submitted:
                # Collect all inputs and convert Yes/No fields to integers for processing
                input_data = {
                    'age': age,
                    'self_employed': 1 if self_employed == 'Yes' else 0,
                    'family_history': 1 if family_history == 'Yes' else 0,
                    'work_interfere': work_interfere,
                    'no_employees': no_employees,
                    'remote_work': 1 if remote_work == 'Yes' else 0,
                    'tech_company': 1 if tech_company == 'Yes' else 0,
                    'leave': leave,
                    'obs_consequence': 1 if obs_consequence == 'Yes' else 0,
                    'gender': gender,
                    'benefits': benefits,
                    'care_options': care_options,
                    'wellness_program': wellness_program,
                    'seek_help': seek_help,
                    'anonymity': anonymity,
                    'mental_health_consequence': mental_health_consequence,
                    'phys_health_consequence': phys_health_consequence,
                    'coworkers': coworkers,
                    'supervisor': supervisor,
                    'mental_health_interview': mental_health_interview,
                    'phys_health_interview': phys_health_interview,
                    'mental_vs_physical': mental_vs_physical,
                }
                
                # Debug print collected data
                if debug_mode:
                    st.subheader("Input Data (Before Processing)")
                    st.json(input_data)
                
                # Preprocess the input data
                processed_data = preprocess_data(input_data)
                
                # Make prediction
                result = predict(processed_data, model, debug=debug_mode)
                
                if result:
                    # Display prediction result
                    st.markdown("---")
                    st.subheader("Prediction Result")
                    
                    # Create a styled box for the result
                    result_color = "rgba(220, 53, 69, 0.2)" if result['treatment_needed'] == 'Yes' else "rgba(40, 167, 69, 0.2)"
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {result_color};">
                        <h3 style="text-align: center;">Treatment Need Prediction: {result['treatment_needed']}</h3>
                        <p style="text-align: center; font-size: 16px;">Based on the provided information, our model predicts that this person 
                        {'may benefit from' if result['treatment_needed'] == 'Yes' else 'may not currently need'} mental health treatment.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display probability meter
                    st.markdown("### Probability")
                    st.progress(result['probability'])
                    st.text(f"Probability of treatment need: {result['probability']:.2%}")
                    
                    # Show full results in debug mode
                    if debug_mode:
                        st.markdown("---")
                        st.subheader("Debug: Full Prediction Details")
                        st.json(result)
                    
                    # Add disclaimer
                    st.markdown("---")
                    st.caption("""
                    **Disclaimer:** This prediction is based on patterns found in survey data and should not be considered a clinical diagnosis. 
                    If you or someone you know is experiencing mental health challenges, please consult with a qualified healthcare professional.
                    """)
    
    with tab2:
        st.header("About MentoPred")
        st.write("""
        MentoPred is a machine learning-based application designed to predict whether an individual might benefit from mental health treatment based on workplace factors and personal attitudes.
        
        ### Problem Statement
        In tech workplaces, mental health is increasingly recognized as crucial yet stigmatized. The goal of this project is to build a model that predicts whether an individual has sought treatment for a mental health condition based on their responses to survey questions about demographics, workplace culture, attitudes, and support systems.
        
        ### Model Information
        This application uses a machine learning model trained on the OSMI Mental Health in Tech Survey dataset. The model analyzes various factors including workplace environment, attitudes toward mental health, and personal history to predict the likelihood that someone would benefit from mental health treatment.
        
        ### Data Privacy
        All data entered in this application is processed locally and is not stored or shared with any third parties.
        """)
        
        st.markdown("---")
        st.write("Created with ❤️ by Xixama")

if __name__ == "__main__":
    main()
