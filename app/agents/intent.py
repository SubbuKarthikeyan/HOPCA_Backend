"""Intent classification and entity extraction agent for hospital operations."""
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from app.agents.state import AgentIntent, AgentState
from app.services.llm.llm_service import LLMService

logger = logging.getLogger(__name__)

# Emergency keywords triggering instant high-priority emergency classification
EMERGENCY_KEYWORDS = [
    "chest pain", "shortness of breath", "heart attack", "unconscious",
    "stroke", "bleeding heavily", "severe trauma", "choking",
    "overdose", "anaphylaxis", "cannot breathe", "seizure", "emergency",
]

# Common hospital departments
DEPARTMENTS = [
    "cardiology", "neurology", "orthopedics", "pediatrics", "oncology",
    "emergency", "radiology", "gastroenterology", "dermatology", "urology",
    "general surgery", "billing", "admissions", "pharmacy", "icu",
]


class IntentClassifierAgent:
    """Classifies user query into hospital intent categories and extracts operational entities."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service if llm_service is not None else LLMService()

    def _extract_rules_and_patterns(self, text: str) -> Tuple[Optional[AgentIntent], float, Dict[str, Any]]:
        """Fast regex and keyword heuristic matching for intent and entities."""
        lower = text.lower()
        entities: Dict[str, Any] = {}

        # 1. Check for Emergency
        for kw in EMERGENCY_KEYWORDS:
            if kw in lower:
                return AgentIntent.EMERGENCY, 0.99, {"emergency_trigger": kw}

        # 2. Extract Doctor Name
        doc_match = re.search(r"(?:dr\.|doctor)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)", text, re.IGNORECASE)
        if doc_match:
            entities["doctor_name"] = f"Dr. {doc_match.group(1).title()}"

        # 3. Extract Department
        for dept in DEPARTMENTS:
            if dept in lower:
                entities["department"] = dept
                break

        # 4. Extract Appointment/Patient IDs
        id_match = re.search(r"\b(apt|pt|doc|pol)-[a-zA-Z0-9-]+\b", text, re.IGNORECASE)
        if id_match:
            entities["entity_id"] = id_match.group(0).upper()

        # 5. Extract date/time mentions
        date_match = re.search(r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", lower)
        if date_match:
            entities["date"] = date_match.group(0)

        time_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", lower)
        if time_match and any(x in lower for x in ["at", "am", "pm", "clock", "morning", "afternoon"]):
            entities["time"] = time_match.group(0)

        # 6. Keyword-based intent disambiguation
        if any(w in lower for w in ["book", "schedule", "reserve", "make an appointment", "set up an appointment"]):
            return AgentIntent.APPOINTMENT_BOOKING, 0.92, entities

        if any(w in lower for w in ["my appointment", "check appointment", "status of appointment", "existing appointment", "cancel appointment", "reschedule"]):
            return AgentIntent.APPOINTMENT_CHECK, 0.90, entities

        if any(w in lower for w in ["available", "availability", "open slots", "when is dr", "free timings", "find appointment", "doctor schedule"]):
            return AgentIntent.APPOINTMENT_INQUIRY, 0.90, entities

        if any(w in lower for w in ["insurance", "copay", "claim", "coverage", "covered by", "bluecross", "aetna", "cigna", "medicare"]):
            return AgentIntent.INSURANCE_INQUIRY, 0.90, entities

        if any(w in lower for w in ["visiting hours", "policy", "visitor", "parking", "cafeteria", "room rate", "wifi", "hospital location", "rules", "guideline"]):
            return AgentIntent.KNOWLEDGE_QUERY, 0.88, entities

        if re.match(r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy|welcome)\b", lower.strip()) or lower.strip() in ["how are you", "thank you", "thanks", "bye", "goodbye"]:
            return AgentIntent.GENERAL_CONVERSATION, 0.95, entities

        return None, 0.0, entities

    async def classify(self, text: str) -> Tuple[AgentIntent, float, Dict[str, Any]]:
        """Classify user intent and extract entities using rule-boosted LLM extraction."""
        # 1. Try heuristic detection first
        fast_intent, confidence, entities = self._extract_rules_and_patterns(text)
        if fast_intent and confidence >= 0.85:
            return fast_intent, confidence, entities

        # 2. LLM Fallback for ambiguous or complex phrasing
        prompt = (
            "You are an intent classifier for a hospital management assistant. "
            "Analyze the user's message and classify it into exactly one intent from:\n"
            "- KNOWLEDGE_QUERY (questions about hospital policies, doctors, departments, visiting hours, amenities)\n"
            "- APPOINTMENT_INQUIRY (asking for available slots/times for doctors or departments)\n"
            "- APPOINTMENT_BOOKING (requesting to book a new appointment)\n"
            "- APPOINTMENT_CHECK (checking, modifying, or querying an existing appointment)\n"
            "- INSURANCE_INQUIRY (questions on insurance networks, coverage, copays)\n"
            "- EMERGENCY (urgent life-threatening conditions, severe acute symptoms)\n"
            "- GENERAL_CONVERSATION (greetings, pleasantries, non-operational talk)\n\n"
            "Return valid JSON only in this exact format:\n"
            '{"intent": "INTENT_NAME", "confidence": 0.95, "entities": {"doctor_name": "...", "department": "...", "date": "...", "time": "..."}}\n\n'
            f'User Message: "{text}"'
        )

        try:
            response_text, _, _ = await self.llm_service.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            # Clean JSON formatting markdown fences if present
            cleaned_json = response_text.strip()
            if cleaned_json.startswith("```"):
                cleaned_json = re.sub(r"^```[a-zA-Z]*\n", "", cleaned_json)
                cleaned_json = re.sub(r"\n```$", "", cleaned_json)

            data = json.loads(cleaned_json)
            intent_str = data.get("intent", "KNOWLEDGE_QUERY").upper()
            intent = AgentIntent[intent_str] if intent_str in AgentIntent.__members__ else AgentIntent.KNOWLEDGE_QUERY
            llm_entities = data.get("entities", {})
            # Merge heuristic and LLM entities
            merged_entities = {**entities, **{k: v for k, v in llm_entities.items() if v}}
            return intent, float(data.get("confidence", 0.85)), merged_entities
        except Exception as exc:
            logger.warning("LLM intent classification failed: %s. Defaulting to KNOWLEDGE_QUERY.", exc)
            return AgentIntent.KNOWLEDGE_QUERY, 0.70, entities

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node execution function for Intent Classification."""
        messages = state.get("messages", [])
        last_message = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_message = m.get("content", "")
                break

        if not last_message:
            return {
                "intent": AgentIntent.GENERAL_CONVERSATION.value,
                "intent_confidence": 1.0,
                "current_step": "intent_classified",
            }

        intent, confidence, entities = await self.classify(last_message)
        return {
            "intent": intent.value,
            "intent_confidence": confidence,
            "entities": entities,
            "current_step": "intent_classified",
        }
