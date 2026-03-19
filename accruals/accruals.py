from pathlib import Path
from dotenv import load_dotenv
import os

# Get project root
project_root = Path(__file__).resolve().parents[1]

env_path = project_root / ".env"

# Debug prints
print("Looking for .env at:", env_path)
print("Exists:", env_path.exists())

# Load .env
load_dotenv(env_path)

print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))