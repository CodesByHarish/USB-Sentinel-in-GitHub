import json
import re
import logging
import httpx
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types

from backend.config import settings
from backend.models import LLMAnalysisResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Business Analyst conducting a live requirement-gathering interview for a Business Requirements Document (BRD).
- Detect the exact code-mixed language style used by the user (e.g., Tanglish, Hinglish, English).
- ALWAYS reply in the exact same language style/dialect. Keep responses brief (1–2 short sentences maximum) so it sounds natural when spoken aloud.
- Maintain state extraction: map user roles to 'stakeholders', product capabilities to 'scope.in', and operational spikes (e.g., Diwali sales) to 'non_functional_reqs' (scalability).
- Return STRICT JSON only:
{
  "updated_fields": {
    "objective": "string or null",
    "stakeholders": ["string"],
    "scope": {
      "in": ["string"],
      "out": ["string"]
    },
    "functional_reqs": ["string"],
    "non_functional_reqs": ["string"],
    "constraints": ["string"],
    "open_questions": ["string"]
  },
  "next_action": "probe" | "next_topic" | "finish",
  "next_question": "Short response in user's spoken dialect",
  "detected_language_code": "ta-IN" | "hi-IN" | "en-IN"
}

Omit empty fields inside updated_fields if nothing was updated for that category. Do not wrap in markdown quotes; output strict JSON only.
"""

class MultiProviderReasoningEngine:
    """
    Reasoning Brain with Multi-Provider Fallback:
    1. Primary Brain: Groq Cloud (llama-3.3-70b-versatile)
    2. Fallback Brain: Google Gemini 2.5 Flash (gemini-2.5-flash)
    """
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None
    ):
        self.groq_api_key = groq_api_key or settings.GROQ_API_KEY
        self.groq_model = groq_model or settings.GROQ_MODEL
        self.groq_api_url = settings.GROQ_API_URL

        self.gemini_api_key = gemini_api_key or settings.GEMINI_API_KEY
        self.gemini_model = gemini_model or settings.GEMINI_MODEL
        self._gemini_client = genai.Client(api_key=self.gemini_api_key)

    def analyze_turn(
        self,
        conversation_history: List[Dict[str, Any]],
        current_brd_state: Dict[str, Any],
        latest_user_input: str
    ) -> LLMAnalysisResult:
        """
        Executes reasoning with Groq Cloud as primary and Gemini 2.5 Flash as fallback.
        """
        prompt = self._build_prompt(conversation_history, current_brd_state, latest_user_input)

        # 1. Try Primary Engine: Groq Cloud
        try:
            raw_json = self._call_groq(prompt)
            result = self._parse_response(raw_json)
            result.active_brain = "groq"
            logger.info("Successfully processed turn via Primary Brain: Groq Cloud")
            return result
        except Exception as groq_err:
            # 2. Automatic Fallback Mechanism: Gemini 2.5 Flash
            logger.warning(
                f"[WARNING] Groq API unavailable or rate-limited ({groq_err}). "
                "Triggering fallback to Gemini 2.5 Flash."
            )
            try:
                raw_json = self._call_gemini(prompt)
                result = self._parse_response(raw_json)
                result.active_brain = "gemini"
                logger.info("Successfully processed turn via Fallback Brain: Gemini 2.5 Flash")
                return result
            except Exception as gemini_err:
                logger.error(f"Both Groq and Gemini failed: {gemini_err}", exc_info=True)
                return LLMAnalysisResult(
                    updated_fields={},
                    next_action="probe",
                    next_question="Could you please tell me more about that?",
                    detected_language_code="en-IN",
                    active_brain="fallback-error"
                )

    def _build_prompt(
        self,
        conversation_history: List[Dict[str, Any]],
        current_brd_state: Dict[str, Any],
        latest_user_input: str
    ) -> str:
        formatted_history = []
        for turn in conversation_history[-10:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            formatted_history.append(f"{role.upper()}: {content}")

        history_str = "\n".join(formatted_history) if formatted_history else "(No prior turns)"

        return f"""
Current BRD State JSON:
{json.dumps(current_brd_state, indent=2)}

Conversation History:
{history_str}

Latest User Input:
"{latest_user_input}"

Analyze the input in the context of the interview and update the BRD state.
Return STRICT JSON only matching the schema.
"""

    def _call_groq(self, prompt: str) -> str:
        """Invokes Groq Cloud API with response_format={'type': 'json_object'}."""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        # Try requested primary model first, fallback to available key model if 404
        candidate_models = [self.groq_model, "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
        last_error = None

        with httpx.Client(timeout=15.0) as client:
            for model_id in candidate_models:
                payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2
                }
                response = client.post(self.groq_api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status_code == 404:
                    # Model not found on this key, try next candidate model
                    last_error = f"Groq model '{model_id}' not found (404)"
                    continue
                else:
                    # 429 rate limit or service error
                    raise RuntimeError(f"Groq API error HTTP {response.status_code}: {response.text}")

        raise RuntimeError(f"Groq API unavailable: {last_error}")

    def _call_gemini(self, prompt: str) -> str:
        """Invokes Google Gemini 2.5 Flash API as fallback."""
        import time
        for attempt in range(2):
            try:
                response = self._gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )
                if response.text:
                    return response.text
            except Exception as e:
                if "429" in str(e) and attempt == 0:
                    logger.warning("Gemini 2.5 Flash hit rate limit. Waiting 3s before retry...")
                    time.sleep(3.0)
                    continue
                raise e
        raise RuntimeError("Gemini fallback failed after retries.")

    def _parse_response(self, raw_text: str) -> LLMAnalysisResult:
        """Parses and sanitizes LLM JSON output from either Groq or Gemini."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned)
            cleaned = re.sub(r"```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except Exception:
                    data = {}
            else:
                data = {}

        # Handle updated_fields (dict or list of strings)
        raw_updated = data.get("updated_fields", {})
        if isinstance(raw_updated, dict):
            updated_fields = raw_updated
        elif isinstance(raw_updated, list):
            # If emitted as list of modified field keys, pull values from root if present
            updated_fields = {}
            for field in raw_updated:
                if isinstance(field, str) and field in data:
                    updated_fields[field] = data[field]
        else:
            updated_fields = {}

        next_action = data.get("next_action", "probe")
        if next_action not in ["probe", "next_topic", "finish"]:
            next_action = "probe"

        next_question = data.get("next_question", "Could you please elaborate?")
        if not isinstance(next_question, str) or not next_question.strip():
            next_question = "Could you tell me more about that?"

        # Extract or detect language code
        detected_lang = data.get("detected_language_code", "").strip().lower()
        if not detected_lang or detected_lang not in ["ta-in", "hi-in", "en-in", "te-in", "bn-in", "kn-in", "ml-in"]:
            q_lower = next_question.lower()
            tanglish_markers = ["inga", "anga", "venum", "panra", "pannala", "irukku", "sema", "ungalukku", "unga", "kooda", "enna", "mattuma", "irukanga", "solreenga", "naan"]
            hinglish_markers = ["karna", "chahiye", "hum", "samajh", "hoga", "hain", "kaun", "kaise", "batao", "kya", "alawa", "bhi"]

            if any(m in q_lower for m in tanglish_markers):
                detected_lang = "ta-IN"
            elif any(m in q_lower for m in hinglish_markers):
                detected_lang = "hi-IN"
            else:
                detected_lang = "en-IN"
        else:
            parts = detected_lang.split("-")
            detected_lang = f"{parts[0].lower()}-{parts[1].upper()}" if len(parts) == 2 else detected_lang

        return LLMAnalysisResult(
            updated_fields=updated_fields,
            next_action=next_action,
            next_question=next_question,
            detected_language_code=detected_lang
        )

reasoning_engine = MultiProviderReasoningEngine()
