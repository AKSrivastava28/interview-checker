import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

dotenv_path = BASE_DIR / ".env"
if not dotenv_path.exists():
    dotenv_path = ROOT_DIR / ".env"

load_dotenv(dotenv_path=dotenv_path)

GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_API_BASE = os.getenv("GROK_API_BASE", "https://api.x.ai/v1")

BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
SESSION_SECRET = os.getenv("SESSION_SECRET", "super_secret_interview_integrity_key")
