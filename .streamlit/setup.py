import streamlit as st
import os
import subprocess
import sys

# Check if we're running on Streamlit Cloud
is_cloud = os.environ.get('STREAMLIT_SHARING') is not None

st.title("MentoPred Setup")

st.write("Setting up your environment...")

# Check if boto3 is installed
try:
    import boto3
    st.success("boto3 is installed")
except ImportError:
    st.warning("boto3 is not installed, attempting to install it...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
        st.success("Successfully installed boto3!")
    except Exception as e:
        st.error(f"Failed to install boto3: {e}")

# Check if model exists
model_path = os.path.join('models', 'final_model.pkl')
if os.path.exists(model_path):
    st.success(f"Model found at {model_path}")
else:
    st.warning(f"Model not found at {model_path}")
    
    # Try to download from S3 if credentials exist
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Get credentials from environment or Streamlit secrets
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        endpoint_url = os.environ.get('B2_ENDPOINT_URL')
        
        # Check if we can get from Streamlit secrets
        if not aws_access_key and hasattr(st, 'secrets'):
            aws_access_key = st.secrets.get("b2remote", {}).get("AWS_ACCESS_KEY_ID")
            aws_secret_key = st.secrets.get("b2remote", {}).get("AWS_SECRET_ACCESS_KEY")
            endpoint_url = st.secrets.get("b2remote", {}).get("B2_ENDPOINT_URL")
        
        if aws_access_key and aws_secret_key:
            st.info("AWS credentials found, attempting to download model...")
            
            # Make sure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Create an S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                endpoint_url=endpoint_url
            )
            
            # Download the file
            s3_client.download_file(
                'dvc-mlops',  # bucket name
                'models/final_model.pkl',  # s3 object path
                model_path  # local file path
            )
            
            st.success("Successfully downloaded model from cloud storage!")
        else:
            st.error("AWS credentials not available")
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")

st.write("Setup complete! You can now run the main app.")
st.button("Refresh")