from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class Scope(BaseModel):
    in_: List[str] = Field(default_factory=list, alias="in")
    out: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True

    def model_dump(self, *args, **kwargs):
        # Always output "in" key instead of "in_"
        data = super().model_dump(*args, **kwargs)
        if "in_" in data:
            data["in"] = data.pop("in_")
        return data

class BRDState(BaseModel):
    objective: Optional[str] = None
    stakeholders: List[str] = Field(default_factory=list)
    scope: Scope = Field(default_factory=Scope)
    functional_reqs: List[str] = Field(default_factory=list)
    non_functional_reqs: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "stakeholders": list(self.stakeholders),
            "scope": {
                "in": list(self.scope.in_),
                "out": list(self.scope.out)
            },
            "functional_reqs": list(self.functional_reqs),
            "non_functional_reqs": list(self.non_functional_reqs),
            "constraints": list(self.constraints),
            "open_questions": list(self.open_questions)
        }

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class Session(BaseModel):
    session_id: str
    brd_state: BRDState = Field(default_factory=BRDState)
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class TurnRequest(BaseModel):
    session_id: Optional[str] = None
    user_text: Optional[str] = None
    user_audio_base64: Optional[str] = None  # Base64 audio for Sarvam Saaras v3 ASR
    target_language_code: Optional[str] = "auto"
    speaker: Optional[str] = "meera"

class LLMAnalysisResult(BaseModel):
    """Output strictly from text reasoning engine (Groq with Gemini Fallback)."""
    updated_fields: Dict[str, Any] = Field(default_factory=dict)
    next_action: Literal["probe", "next_topic", "finish"]
    next_question: str
    detected_language_code: str = "hi-IN"
    active_brain: str = "groq"

class TurnResponse(BaseModel):
    session_id: str
    next_action: Literal["probe", "next_topic", "finish"]
    next_question: str
    updated_fields: Dict[str, Any]
    brd_state: Dict[str, Any]
    conversation_history: List[ConversationTurn]
    # Voice audio generated strictly by Sarvam Bulbul v3
    audio_base64: Optional[str] = None
    audio_format: str = "audio/wav"
    detected_language_code: str = "hi-IN"
    # Execution Chain Metadata
    active_brain: str = "groq"
    transcribed_text: Optional[str] = None

class TTSRequest(BaseModel):
    text: str
    target_language_code: Optional[str] = "hi-IN"
    speaker: Optional[str] = "meera"

class TTSResponse(BaseModel):
    audio_base64: Optional[str] = None
    audio_format: str = "audio/wav"
