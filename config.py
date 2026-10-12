import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_pages"
FAISS_DIR = DATA_DIR / "faiss_index"
CHUNKS_FILE = DATA_DIR / "chunks.json"
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"

for directory in [DATA_DIR, RAW_DATA_DIR, FAISS_DIR, AUDIO_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# DeepSeek Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Server Configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Multilingual Configuration (6 Supported Languages: English, Hindi, Kannada, Telugu, Tamil, Gujarati)
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "native": "English",
        "voice_code": "en-IN",
        "gtts_code": "en",
        "flag": "🇬🇧",
        "greeting": "Hello! I am your DISH TV AI Assistant. How can I assist you with your DISH TV recharge, channels, error codes, or packs today?"
    },
    "hi": {
        "name": "Hindi",
        "native": "हिन्दी",
        "voice_code": "hi-IN",
        "gtts_code": "hi",
        "flag": "🇮🇳",
        "greeting": "नमस्ते! मैं आपका DISH TV AI सहायक हूँ। आज मैं आपकी डिश टीवी सेवा, एरर कोड, रिचार्ज या पैक से संबंधित कैसे मदद कर सकता हूँ?"
    },
    "kn": {
        "name": "Kannada",
        "native": "ಕನ್ನಡ",
        "voice_code": "kn-IN",
        "gtts_code": "kn",
        "flag": "🇮🇳",
        "greeting": "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ DISH TV AI ಸಹಾಯಕ. ನಿಮ್ಮ DISH TV ರೀಚಾರ್ಜ್, ದೋಷ ಕೋಡ್‌ಗಳು ಅಥವಾ ಪ್ಯಾಕ್‌ಗಳ ಕುರಿತು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?"
    },
    "te": {
        "name": "Telugu",
        "native": "తెలుగు",
        "voice_code": "te-IN",
        "gtts_code": "te",
        "flag": "🇮🇳",
        "greeting": "నమస్కారం! నేను మీ DISH TV AI అసిస్టెంట్‌ని. మీ DISH TV రీఛార్జ్, ఎర్రర్ కోడ్‌లు లేదా ప్యాక్‌ల గురించి మీకు ఎలా సహాయపడగలను?"
    },
    "ta": {
        "name": "Tamil",
        "native": "தமிழ்",
        "voice_code": "ta-IN",
        "gtts_code": "ta",
        "flag": "🇮🇳",
        "greeting": "வணக்கம்! நான் உங்கள் DISH TV AI உதவியாளர். உங்கள் DISH TV ரீசார்ஜ், பிழை குறியீடுகள் அல்லது பேக்குகள் பற்றி உங்களுக்கு நான் எவ்வாறு உதವ முடியும்?"
    },
    "gu": {
        "name": "Gujarati",
        "native": "ગુજરાતી",
        "voice_code": "gu-IN",
        "gtts_code": "gu",
        "flag": "🇮🇳",
        "greeting": "નમસ્તે! હું તમારો DISH TV AI સહાયક છું. આજે હું તમને DISH TV રિચાર્જ, એરર કોડ્સ અથવા ચેનલ પેક સંબંધિત કેવી રીતે મદદ કરી શકું?"
    }
}

