import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel
    class BaseSettings(BaseModel):
        pass

class Settings(BaseSettings):
    # Primary Reasoning Brain: Groq Cloud
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_API_URL: str = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

    # Fallback Reasoning Brain: Google Gemini 2.5 Flash
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Complete Sarvam AI Voice Stack
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    SARVAM_STT_URL: str = os.getenv(
        "SARVAM_STT_URL",
        "https://api.sarvam.ai/speech-to-text"
    )
    SARVAM_TTS_URL: str = os.getenv(
        "SARVAM_TTS_URL",
        "https://api.sarvam.ai/text-to-speech"
    )
    SARVAM_DEFAULT_SPEAKER: str = os.getenv("SARVAM_DEFAULT_SPEAKER", "meera")
    SARVAM_DEFAULT_LANGUAGE: str = os.getenv("SARVAM_DEFAULT_LANGUAGE", "hi-IN")
    SARVAM_TTS_MODEL: str = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
    SARVAM_STT_MODEL: str = os.getenv("SARVAM_STT_MODEL", "saaras:v3")

    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()
