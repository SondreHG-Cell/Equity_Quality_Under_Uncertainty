from pathlib import Path
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load .env from project root
project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")

# Retrieve API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)