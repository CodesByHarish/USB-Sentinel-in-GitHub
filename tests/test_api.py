import io
import unittest
from fastapi.testclient import TestClient
from backend.main import app
from backend.models import BRDState, LLMAnalysisResult
from backend.session_store import session_store
from backend.voice_service import voice_service
from backend.reasoning_engine import reasoning_engine

client = TestClient(app)

class TestBRDBackend(unittest.TestCase):
    def test_initial_brd_state_schema(self):
        state = BRDState()
        d = state.to_dict()
        self.assertEqual(d, {
            "objective": None,
            "stakeholders": [],
            "scope": {"in": [], "out": []},
            "functional_reqs": [],
            "non_functional_reqs": [],
            "constraints": [],
            "open_questions": []
        })

    def test_session_store_merge(self):
        session = session_store.create_session("test-session-multi")
        self.assertIsNone(session.brd_state.objective)

        updates = {
            "objective": "Build modern B2B SaaS billing engine",
            "stakeholders": ["Billing Admin", "End Users"],
            "scope": {
                "in": ["Stripe integration", "Invoice generation"],
                "out": ["Cryptocurrency payments"]
            },
            "functional_reqs": ["Support prorated subscription changes"],
            "non_functional_reqs": ["99.99% billing engine uptime"],
            "constraints": ["Must comply with PCI-DSS Level 1"],
            "open_questions": ["What is the initial target currency?"]
        }
        session_store.merge_brd_updates(session, updates)

        state = session.brd_state.to_dict()
        self.assertEqual(state["objective"], "Build modern B2B SaaS billing engine")
        self.assertIn("Billing Admin", state["stakeholders"])
        self.assertIn("Stripe integration", state["scope"]["in"])
        self.assertIn("Cryptocurrency payments", state["scope"]["out"])
        self.assertIn("Support prorated subscription changes", state["functional_reqs"])
        self.assertIn("99.99% billing engine uptime", state["non_functional_reqs"])
        self.assertIn("Must comply with PCI-DSS Level 1", state["constraints"])
        self.assertIn("What is the initial target currency?", state["open_questions"])

    def test_separation_of_concerns_llm_result_has_no_audio(self):
        """Verify LLMAnalysisResult schema contains strictly text reasoning fields."""
        fields = LLMAnalysisResult.model_fields.keys()
        self.assertIn("updated_fields", fields)
        self.assertIn("next_action", fields)
        self.assertIn("next_question", fields)
        self.assertNotIn("audio", fields)
        self.assertNotIn("audio_base64", fields)

    def test_sarvam_voice_services_tts_and_asr(self):
        """Verify Sarvam Bulbul v3 TTS and Saaras v3 ASR integration."""
        # 1. Synthesize audio with Bulbul
        test_text = "Namaste, Sarvam voice stack verification."
        audio_b64 = voice_service.synthesize_speech(
            text=test_text,
            target_language_code="hi-IN",
            speaker="meera"
        )
        self.assertIsNotNone(audio_b64)
        self.assertIsInstance(audio_b64, str)
        self.assertGreater(len(audio_b64), 1000)

        # 2. Transcribe audio with Saaras
        import base64
        audio_bytes = base64.b64decode(audio_b64)
        asr_res = voice_service.transcribe_audio(audio_bytes, filename="test.wav")
        self.assertIsNotNone(asr_res)
        self.assertIn("transcript", asr_res)

    def test_multi_provider_reasoning_and_fallback(self):
        """Verify primary engine (Groq) and fallback mechanism (Gemini)."""
        history = []
        state = BRDState().to_dict()
        user_input = "We need an inventory tracking app for warehouse supervisors."

        # Analyze turn triggers Groq primary or fallback to Gemini
        res = reasoning_engine.analyze_turn(history, state, user_input)
        self.assertIn(res.active_brain, ["groq", "gemini"])
        self.assertIn(res.next_action, ["probe", "next_topic", "finish"])
        self.assertTrue(len(res.next_question) > 0)

    def test_execution_chain_turn_api(self):
        """Verify full execution chain: Groq Brain (Gemini Fallback) -> Sarvam Bulbul TTS."""
        payload = {
            "user_text": "Naan oru small business vechurukken, products sell pandren, aana en customers order pandradhukku app venum.",
            "target_language_code": "auto",
            "speaker": "meera"
        }
        response = client.post("/api/interview/turn", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check brain output
        self.assertIn("session_id", data)
        self.assertIn(data.get("active_brain"), ["groq", "gemini"])
        self.assertIn(data["next_action"], ["probe", "next_topic", "finish"])
        self.assertTrue(len(data["next_question"]) > 0)

        # Check Sarvam Bulbul TTS output
        self.assertIsNotNone(data.get("audio_base64"))
        self.assertGreater(len(data.get("audio_base64")), 1000)

        # Check export
        export_resp = client.get(f"/api/interview/export/{data['session_id']}")
        self.assertEqual(export_resp.status_code, 200)
        self.assertIn("Business Requirements Document (BRD)", export_resp.text)

if __name__ == "__main__":
    unittest.main()
