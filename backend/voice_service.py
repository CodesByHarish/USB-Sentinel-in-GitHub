import logging
import io
import httpx
from typing import Optional, Dict, Any
from backend.config import settings

logger = logging.getLogger(__name__)

# Valid speaker mappings for Sarvam Bulbul v3
# In bulbul:v3, 'meera' maps to 'ritu' (premier female Indian voice)
V3_SPEAKER_FALLBACKS = {
    "meera": "ritu",
}

class SarvamVoiceService:
    """
    Dedicated voice service handling all audio listening & speaking layers:
    1. Voice Input (ASR): Sarvam Saaras v3 (transcribe user audio)
    2. Voice Output (TTS): Sarvam Bulbul v3 (synthesize agent questions)
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        stt_url: Optional[str] = None,
        tts_url: Optional[str] = None,
        default_speaker: Optional[str] = None,
        default_language: Optional[str] = None
    ):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.stt_url = stt_url or settings.SARVAM_STT_URL
        self.tts_url = tts_url or settings.SARVAM_TTS_URL
        self.default_speaker = default_speaker or settings.SARVAM_DEFAULT_SPEAKER
        self.default_language = default_language or settings.SARVAM_DEFAULT_LANGUAGE
        self.tts_model = settings.SARVAM_TTS_MODEL
        self.stt_model = settings.SARVAM_STT_MODEL

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: str = "unknown"
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribes user voice input using Sarvam Saaras v3 ASR.
        Returns dict with 'transcript', 'language_code', or None on failure.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            return None

        headers = {
            "api-subscription-key": self.api_key
        }

        # Infer content type
        content_type = "audio/wav"
        if filename.endswith(".webm"):
            content_type = "audio/webm"
        elif filename.endswith(".mp3"):
            content_type = "audio/mpeg"
        elif filename.endswith(".ogg"):
            content_type = "audio/ogg"

        files = {
            "file": (filename, io.BytesIO(audio_bytes), content_type)
        }
        data = {
            "model": self.stt_model,
            "language_code": language_code
        }

        try:
            with httpx.Client(timeout=25.0) as client:
                response = client.post(self.stt_url, headers=headers, files=files, data=data)
                if response.status_code == 200:
                    res_json = response.json()
                    transcript = res_json.get("transcript", "").strip()
                    detected_lang = res_json.get("language_code", "unknown")
                    logger.info(f"Sarvam Saaras transcribed audio successfully: '{transcript}' (lang: {detected_lang})")
                    return {
                        "transcript": transcript,
                        "language_code": detected_lang
                    }
                else:
                    logger.error(f"Sarvam Saaras ASR error ({response.status_code}): {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Exception during Sarvam Saaras transcription: {e}", exc_info=True)
            return None

    def synthesize_speech(
        self,
        text: str,
        target_language_code: Optional[str] = None,
        speaker: Optional[str] = None,
        pitch: float = 0.0,
        pace: float = 1.05,
        loudness: float = 1.5,
        speech_sample_rate: int = 24000
    ) -> Optional[str]:
        """
        Synthesizes speech using Sarvam Bulbul v3 TTS.
        Returns base64-encoded audio string, or None on failure.
        """
        if not text or not text.strip():
            return None

        clean_text = text.strip()
        lang_code = target_language_code or self.default_language
        spk = speaker or self.default_speaker

        # Resolve speaker for bulbul:v3
        active_speaker = V3_SPEAKER_FALLBACKS.get(spk.lower(), spk)

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload: Dict[str, Any] = {
            "inputs": [clean_text],
            "target_language_code": lang_code,
            "speaker": active_speaker,
            "pitch": pitch,
            "pace": pace,
            "loudness": loudness,
            "speech_sample_rate": speech_sample_rate,
            "enable_preprocessing": True,
            "model": self.tts_model
        }

        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.post(self.tts_url, headers=headers, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    audios = data.get("audios", [])
                    if audios and len(audios) > 0:
                        logger.info(f"Sarvam Bulbul v3 synthesized speech successfully ({len(audios[0])} chars).")
                        return audios[0]
                    else:
                        logger.warning(f"Sarvam TTS returned empty audio list: {data}")
                        return None
                else:
                    # Retry with 'ritu' if speaker failed
                    if response.status_code == 400 and active_speaker != "ritu":
                        logger.warning(f"Retrying Sarvam TTS with speaker 'ritu' due to: {response.text}")
                        payload["speaker"] = "ritu"
                        retry_resp = client.post(self.tts_url, headers=headers, json=payload)
                        if retry_resp.status_code == 200:
                            data = retry_resp.json()
                            audios = data.get("audios", [])
                            if audios:
                                return audios[0]

                    logger.error(f"Sarvam TTS API error ({response.status_code}): {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Exception during Sarvam TTS synthesis: {e}", exc_info=True)
            return None

voice_service = SarvamVoiceService()
