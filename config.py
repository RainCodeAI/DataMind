# config.py

import os
from dotenv import load_dotenv

# Load environment variables from the .env file if it exists
load_dotenv()

def get_openai_api_key():
    """
    Fetches the OpenAI API key from environment variables.
    
    Returns:
        str: The OpenAI API key
        
    Raises:
        ValueError: If the API key is not found
    """
    # Try to get the API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set your OpenAI API key as an environment variable or in a .env file."
        )
    
    return api_key

def validate_environment():
    """
    Validates that all required environment variables are set.
    
    Returns:
        bool: True if all required variables are set, False otherwise
    """
    try:
        get_openai_api_key()
        return True
    except ValueError:
        return False
