import uuid
import threading
from typing import Dict, Optional, Any, List
from datetime import datetime
from backend.models import Session, BRDState, Scope, ConversationTurn

def _dedup_append(existing_list: List[str], new_items: Any) -> None:
    if isinstance(new_items, str):
        items = [new_items]
    elif isinstance(new_items, list):
        items = new_items
    else:
        return

    lower_existing = {item.strip().lower() for item in existing_list if isinstance(item, str)}
    for item in items:
        if isinstance(item, str) and item.strip():
            cleaned = item.strip()
            if cleaned.lower() not in lower_existing:
                existing_list.append(cleaned)
                lower_existing.add(cleaned.lower())

class InMemorySessionStore:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()

    def create_session(self, session_id: Optional[str] = None) -> Session:
        with self._lock:
            sid = session_id or str(uuid.uuid4())
            session = Session(session_id=sid, brd_state=BRDState())
            self._sessions[sid] = session
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            return self.create_session(session_id)

    def add_turn(self, session: Session, role: str, content: str) -> None:
        with self._lock:
            turn = ConversationTurn(role=role, content=content)
            session.conversation_history.append(turn)
            session.updated_at = datetime.utcnow().isoformat()

    def merge_brd_updates(self, session: Session, updated_fields: Dict[str, Any]) -> BRDState:
        with self._lock:
            if not updated_fields or not isinstance(updated_fields, dict):
                return session.brd_state

            state = session.brd_state

            # 1. Objective
            if "objective" in updated_fields and updated_fields["objective"]:
                obj_val = updated_fields["objective"]
                if isinstance(obj_val, str) and obj_val.strip():
                    state.objective = obj_val.strip()

            # 2. Stakeholders
            if "stakeholders" in updated_fields:
                _dedup_append(state.stakeholders, updated_fields["stakeholders"])

            # 3. Scope
            if "scope" in updated_fields and isinstance(updated_fields["scope"], dict):
                scope_update = updated_fields["scope"]
                if "in" in scope_update:
                    _dedup_append(state.scope.in_, scope_update["in"])
                if "out" in scope_update:
                    _dedup_append(state.scope.out, scope_update["out"])

            # 4. Functional Requirements
            if "functional_reqs" in updated_fields:
                _dedup_append(state.functional_reqs, updated_fields["functional_reqs"])

            # 5. Non-functional Requirements
            if "non_functional_reqs" in updated_fields:
                _dedup_append(state.non_functional_reqs, updated_fields["non_functional_reqs"])

            # 6. Constraints
            if "constraints" in updated_fields:
                _dedup_append(state.constraints, updated_fields["constraints"])

            # 7. Open Questions
            if "open_questions" in updated_fields:
                _dedup_append(state.open_questions, updated_fields["open_questions"])

            session.updated_at = datetime.utcnow().isoformat()
            return state

    def reset_session(self, session_id: str) -> Session:
        with self._lock:
            new_session = Session(session_id=session_id, brd_state=BRDState())
            self._sessions[session_id] = new_session
            return new_session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

session_store = InMemorySessionStore()
