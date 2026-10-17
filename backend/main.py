import os
import base64
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse

from backend.config import settings
from backend.models import TurnRequest, TurnResponse, Session, TTSRequest, TTSResponse
from backend.session_store import session_store
from backend.reasoning_engine import reasoning_engine
from backend.voice_service import voice_service

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Live Voice-Based BRD Interview Agent",
    description="Multi-Provider Fallback (Groq + Gemini) & Sarvam AI Voice Stack (Saaras ASR + Bulbul TTS)",
    version="2.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _process_interview_turn(
    session_id: Optional[str],
    user_text: str,
    target_language_code: Optional[str] = "auto",
    speaker: Optional[str] = "meera",
    transcribed_text: Optional[str] = None
) -> TurnResponse:
    """
    Executes the complete end-to-end interview turn:
    1. Session & history update
    2. Reasoning Brain: Groq Cloud primary with automatic Gemini 2.5 Flash fallback
    3. State extraction & merging
    4. Voice Output: Sarvam Bulbul v3 TTS using detected dialect
    """
    # 1. Retrieve or create session
    session = session_store.get_or_create_session(session_id)

    # 2. Add user turn to conversation history
    session_store.add_turn(session, role="user", content=user_text.strip())

    # 3. Prepare context payload for Reasoning Brain
    history_dicts = [
        {"role": turn.role, "content": turn.content}
        for turn in session.conversation_history
    ]
    current_state_dict = session.brd_state.to_dict()

    # 4. Multi-Provider Reasoning Brain (Groq Cloud -> Gemini Fallback)
    analysis_result = reasoning_engine.analyze_turn(
        conversation_history=history_dicts,
        current_brd_state=current_state_dict,
        latest_user_input=user_text.strip()
    )

    # 5. Merge detected updates into in-memory BRD state
    session_store.merge_brd_updates(session, analysis_result.updated_fields)

    # 6. Record assistant's question in conversation history
    session_store.add_turn(session, role="assistant", content=analysis_result.next_question)

    # 7. Voice Output: Sarvam Bulbul v3 TTS
    effective_lang = target_language_code
    if not effective_lang or effective_lang.lower() in ["auto", "default", "none"]:
        effective_lang = analysis_result.detected_language_code

    audio_base64 = voice_service.synthesize_speech(
        text=analysis_result.next_question,
        target_language_code=effective_lang,
        speaker=speaker
    )

    return TurnResponse(
        session_id=session.session_id,
        next_action=analysis_result.next_action,
        next_question=analysis_result.next_question,
        updated_fields=analysis_result.updated_fields,
        brd_state=session.brd_state.to_dict(),
        conversation_history=session.conversation_history,
        audio_base64=audio_base64,
        audio_format="audio/wav",
        detected_language_code=analysis_result.detected_language_code,
        active_brain=getattr(analysis_result, "active_brain", "groq"),
        transcribed_text=transcribed_text
    )

@app.post("/api/interview/turn", response_model=TurnResponse)
def handle_interview_turn(payload: TurnRequest):
    """
    Handles turn via JSON payload:
    - If user_audio_base64 provided -> Transcribes via Sarvam Saaras v3 ASR first.
    - Runs Reasoning Brain (Groq with Gemini Fallback).
    - Synthesizes via Sarvam Bulbul v3 TTS.
    """
    user_text = payload.user_text or ""
    transcribed_text = None

    # Step 1: Voice Input (Sarvam Saaras ASR) if audio supplied
    if payload.user_audio_base64 and payload.user_audio_base64.strip():
        try:
            audio_bytes = base64.b64decode(payload.user_audio_base64)
            asr_res = voice_service.transcribe_audio(audio_bytes, filename="input.wav")
            if asr_res and asr_res.get("transcript"):
                transcribed_text = asr_res["transcript"]
                user_text = transcribed_text
        except Exception as e:
            logger.error(f"Error decoding/transcribing user audio: {e}")

    if not user_text.strip():
        raise HTTPException(status_code=400, detail="Either user_text or user_audio_base64 must be provided.")

    return _process_interview_turn(
        session_id=payload.session_id,
        user_text=user_text,
        target_language_code=payload.target_language_code,
        speaker=payload.speaker,
        transcribed_text=transcribed_text
    )

@app.post("/api/interview/turn/audio", response_model=TurnResponse)
async def handle_interview_turn_audio(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    target_language_code: Optional[str] = Form("auto"),
    speaker: Optional[str] = Form("meera")
):
    """
    Direct audio upload endpoint from browser microphone:
    1. Transcribes speech using Sarvam Saaras v3 ASR
    2. Sends text to Groq Cloud (with Gemini Fallback)
    3. Synthesizes response via Sarvam Bulbul v3 TTS
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    asr_result = voice_service.transcribe_audio(
        audio_bytes=audio_bytes,
        filename=file.filename or "input.wav"
    )

    if not asr_result or not asr_result.get("transcript"):
        raise HTTPException(status_code=422, detail="Sarvam Saaras could not transcribe the audio. Please speak clearly or type.")

    transcribed_text = asr_result["transcript"]
    return _process_interview_turn(
        session_id=session_id,
        user_text=transcribed_text,
        target_language_code=target_language_code,
        speaker=speaker,
        transcribed_text=transcribed_text
    )

@app.post("/api/voice/tts", response_model=TTSResponse)
def synthesize_tts(payload: TTSRequest):
    """
    Standalone TTS endpoint powered strictly by Sarvam Bulbul v3.
    """
    audio_base64 = voice_service.synthesize_speech(
        text=payload.text,
        target_language_code=payload.target_language_code,
        speaker=payload.speaker
    )
    if not audio_base64:
        raise HTTPException(status_code=500, detail="Sarvam TTS synthesis failed.")
    return TTSResponse(audio_base64=audio_base64, audio_format="audio/wav")

@app.get("/api/interview/session/{session_id}")
def get_session(session_id: str):
    """Returns the current state and history of an existing session."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return {
        "session_id": session.session_id,
        "brd_state": session.brd_state.to_dict(),
        "conversation_history": session.conversation_history,
        "created_at": session.created_at,
        "updated_at": session.updated_at
    }

@app.post("/api/interview/reset")
def reset_session(session_id: str = None):
    """Resets or creates a clean session."""
    session = session_store.get_or_create_session(session_id)
    reset_s = session_store.reset_session(session.session_id)
    return {
        "message": "Session reset successfully",
        "session_id": reset_s.session_id,
        "brd_state": reset_s.brd_state.to_dict()
    }

@app.get("/api/interview/export/{session_id}")
def export_brd(session_id: str):
    """Generates a formatted Markdown Business Requirements Document."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    state = session.brd_state.to_dict()
    md_lines = [
        f"# Business Requirements Document (BRD)",
        f"*Generated by Live Voice BRD Interview Agent | Session: `{session_id}`*",
        f"\n---\n",
        f"## 1. Executive Summary & Objective",
        f"{state.get('objective') or '*(Pending definition)*'}\n",
        f"## 2. Key Stakeholders",
    ]
    if state.get("stakeholders"):
        for s in state["stakeholders"]:
            md_lines.append(f"- {s}")
    else:
        md_lines.append("*(No stakeholders identified yet)*")

    md_lines.append("\n## 3. Scope")
    md_lines.append("### In-Scope")
    if state.get("scope", {}).get("in"):
        for item in state["scope"]["in"]:
            md_lines.append(f"- [x] {item}")
    else:
        md_lines.append("*(In-scope boundaries pending)*")

    md_lines.append("\n### Out-of-Scope")
    if state.get("scope", {}).get("out"):
        for item in state["scope"]["out"]:
            md_lines.append(f"- [ ] {item}")
    else:
        md_lines.append("*(Out-of-scope boundaries pending)*")

    md_lines.append("\n## 4. Functional Requirements")
    if state.get("functional_reqs"):
        for idx, item in enumerate(state["functional_reqs"], start=1):
            md_lines.append(f"**FR-{idx:02d}**: {item}")
    else:
        md_lines.append("*(Functional requirements pending)*")

    md_lines.append("\n## 5. Non-Functional Requirements (Scalability, Security, SLA)")
    if state.get("non_functional_reqs"):
        for idx, item in enumerate(state["non_functional_reqs"], start=1):
            md_lines.append(f"**NFR-{idx:02d}**: {item}")
    else:
        md_lines.append("*(Non-functional requirements pending)*")

    md_lines.append("\n## 6. Constraints & Dependencies")
    if state.get("constraints"):
        for item in state["constraints"]:
            md_lines.append(f"- {item}")
    else:
        md_lines.append("*(No specific constraints noted)*")

    md_lines.append("\n## 7. Open Questions & Action Items")
    if state.get("open_questions"):
        for item in state["open_questions"]:
            md_lines.append(f"- [ ] {item}")
    else:
        md_lines.append("*(All questions resolved)*")

    return PlainTextResponse(content="\n".join(md_lines), media_type="text/markdown")

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Live Voice-Based BRD Interview Agent API running."}
