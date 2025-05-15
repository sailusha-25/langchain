import streamlit as st
import os

def load_huggingface_token():
    """
    Load Hugging Face API token from Streamlit secrets or environment variables.
    
    Returns:
        str or None: The Hugging Face API token if available, None otherwise.
    """
    # Try to get token from Streamlit secrets
    if 'huggingface' in st.secrets:
        return st.secrets['huggingface']['api_token']
    
    # Fallback to environment variable
    elif 'HF_API_TOKEN' in os.environ:
        return os.environ['HF_API_TOKEN']
    
    # No token found
    else:
        return None
        
def setup_api_access():
    """
    Set up API access for Hugging Face.
    If token is available, sets it as an environment variable.
    
    Returns:
        bool: True if setup was successful, False otherwise.
    """
    token = load_huggingface_token()
    
    if token:
        # Set environment variable for Hugging Face libraries
        os.environ['HUGGINGFACE_API_TOKEN'] = token
        os.environ['HF_API_TOKEN'] = token  # For compatibility with different libraries
        
        # For libraries that use transformers directly
        try:
            from huggingface_hub import login
            login(token=token)
        except Exception as e:
            st.warning(f"Failed to login to Hugging Face Hub: {e}")
            # Continue execution as the token is still set in env vars
        
        return True
    else:
        return False
