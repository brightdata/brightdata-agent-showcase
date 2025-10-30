"""Configuration settings for Amazon Analytics."""
import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values

# Environment variable loading with security isolation
load_dotenv()

_local_env_path = Path.cwd() / ".env"
_local_env = dotenv_values(_local_env_path) if _local_env_path.exists() else {}

BRIGHT_DATA_TOKEN = os.getenv("BRIGHT_DATA_TOKEN")
# Gemini API key with fallback to environment variable  
GOOGLE_API_KEY = _local_env.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
# Bright Data dataset identifier for Amazon product scraping
BRIGHT_DATA_DATASET_ID = "gd_lwdb4vjm1ehb499uxs"

# Global Amazon marketplace coverage - 23 active marketplaces (verified September 2025)
AMAZON_COUNTRIES = {
    "United States": "https://www.amazon.com",
    "Canada": "https://www.amazon.ca",
    "Mexico": "https://www.amazon.com.mx",
    "Brazil": "https://www.amazon.com.br",
    "United Kingdom": "https://www.amazon.co.uk",
    "Ireland": "https://www.amazon.ie",
    "Germany": "https://www.amazon.de",
    "France": "https://www.amazon.fr",
    "Italy": "https://www.amazon.it",
    "Spain": "https://www.amazon.es",
    "Netherlands": "https://www.amazon.nl",
    "Sweden": "https://www.amazon.se",
    "Poland": "https://www.amazon.pl",
    "Belgium": "https://www.amazon.com.be",
    "Turkey": "https://www.amazon.com.tr",
    "United Arab Emirates": "https://www.amazon.ae",
    "Saudi Arabia": "https://www.amazon.sa",
    "Egypt": "https://www.amazon.eg",
    "South Africa": "https://www.amazon.co.za",
    "India": "https://www.amazon.in",
    "Japan": "https://www.amazon.co.jp",
    "Singapore": "https://www.amazon.sg",
    "Australia": "https://www.amazon.com.au"
}

