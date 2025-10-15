# MentoPred Streamlit Deployment Guide

This guide explains how to properly deploy the MentoPred application on Streamlit Cloud.

## Prerequisites

1. A Streamlit Cloud account
2. Access to the MentoPred GitHub repository
3. Access to the model storage (Backblaze B2 or other S3-compatible storage)

## Steps to Deploy

1. **Fork or push the repository to GitHub** if you haven't already.

2. **Create a new app in Streamlit Cloud**:
   - Connect to your GitHub repository
   - Set the main file path to `app.py`

3. **Configure Secrets**:
   - In the Streamlit Cloud dashboard, go to your app settings
   - Add the following secrets:

```toml
[b2remote]
AWS_ACCESS_KEY_ID = "your_access_key_here"
AWS_SECRET_ACCESS_KEY = "your_secret_key_here"
B2_ENDPOINT_URL = "https://s3.eu-central-003.backblazeb2.com"
```

4. **Deploy the app**:
   - Streamlit Cloud will automatically install the required dependencies from `requirements.txt`
   - The app will attempt to download the model from the configured S3 bucket
   - If the model cannot be downloaded, a demo model will be used instead

## Troubleshooting

- If you see "boto3 not installed" errors, check that your `requirements.txt` file includes boto3
- If you see "Model file not found" errors, verify your S3 credentials in the secrets settings
- If you see "Pipeline is not fitted yet" errors, the model file might be corrupted or the demo model might not be properly fitted

## Testing Locally

To test the deployment setup locally:

1. Install the dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`

## Contact

For support, please contact the repository maintainer.