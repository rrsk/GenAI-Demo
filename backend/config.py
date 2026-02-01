"""
Configuration settings for WellnessAI backend

Environment Variables:
    WEATHERAPI_KEY - Required for weather data
    WELLNESS_LLM_MODEL - Local LLM model choice (see options below)
    USE_LOCAL_LLM - Set to "false" to disable local LLM
    OPENAI_API_KEY - Optional, for API fallback
    ANTHROPIC_API_KEY - Optional, for API fallback

Available LLM Models:
    MEDICAL DOMAIN (Recommended):
        - biogpt: Microsoft BioGPT - 1.5B, PubMed-trained (default)
        - biogpt-medtext: Quantized BioGPT - ~661MB, very fast
        - biomedlm: Stanford BioMedLM - 2.7B, highest accuracy
    
    GENERAL PURPOSE:
        - tinyllama: TinyLlama 1.1B - fast, CPU-friendly
        - phi2: Microsoft Phi-2 - 2.7B, high quality
        - flan-t5: Google Flan-T5-base - efficient
        - mistral: Mistral 7B - best quality, needs GPU
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Whoop Data"
WHOOP_DATA_PATH = DATA_DIR / "whoop_fitness_dataset_100k.csv"
USER_DATA_DIR = BASE_DIR / "user_data"

# Model cache directory
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ============================================
# Local LLM Settings (Primary)
# ============================================
# Default: tinyllama (best for conversational chat)
# For medical terminology: biogpt, biogpt-medtext, biomedlm
# General models: tinyllama, phi2, flan-t5, mistral
LOCAL_LLM_MODEL = os.getenv("WELLNESS_LLM_MODEL", "tinyllama")

# Whether to use local LLM (default True)
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"

# Preload model on startup (slower start, faster first response)
PRELOAD_LLM = os.getenv("PRELOAD_LLM", "false").lower() == "true"

# State-of-union: BioGPT grounds context from user stats, TinyLlama gives the suggestion
# When True: (1) BioGPT produces evidence-based prompt from health context + query
#            (2) TinyLlama takes that prompt + query and returns user-facing response
USE_STATE_OF_UNION = os.getenv("USE_STATE_OF_UNION", "true").lower() == "true"
# Model used for grounding (evidence/research framing). Should be medical: biogpt, biogpt-medtext, biomedlm
WELLNESS_GROUNDING_MODEL = os.getenv("WELLNESS_GROUNDING_MODEL", "biogpt")

# ============================================
# External API Keys (Optional Fallback)
# ============================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# External AI Model settings (fallback only)
DEFAULT_AI_PROVIDER = "local"  # "local", "openai", or "anthropic"
OPENAI_MODEL = "gpt-4-turbo-preview"
ANTHROPIC_MODEL = "claude-3-opus-20240229"

# ============================================
# LSTM Health Forecasting Settings
# ============================================
LSTM_MODEL_PATH = MODELS_DIR / "lstm_health.pt"
LSTM_SCALERS_PATH = MODELS_DIR / "lstm_scalers.joblib"
LSTM_SEQUENCE_LENGTH = 21  # Days of history required
LSTM_FORECAST_HORIZON = 7  # Days to forecast
LSTM_INPUT_FEATURES = 8  # Number of input features
USE_LSTM = os.getenv("USE_LSTM", "true").lower() == "true"
