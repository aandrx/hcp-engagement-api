import os
from dotenv import load_dotenv

load_dotenv()

print(f"GROQ_API_KEY loaded: {bool(os.getenv('GROQ_API_KEY'))}")
if os.getenv('GROQ_API_KEY'):
    print(f"Key starts with: {os.getenv('GROQ_API_KEY')[:10]}...")
