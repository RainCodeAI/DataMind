# the_analyst/config.py

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def get_openai_api_key():
    """Fetches the OpenAI API key from environment variables."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not found in .env file or is not set.")
    return key