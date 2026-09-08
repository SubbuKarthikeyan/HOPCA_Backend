"""Decision and validation node for parameter checking and routing authorization."""
import logging
from typing import Any, Dict, List

from app.agents.state import AgentIntent, AgentState, ValidationStatus

logger = logging.getLogger(__name__)


class DecisionValidationNode:
    """Validates operational requests for required parameters and determines workflow routing."""



    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Validate state parameters before tool execution."""
        intent_str = state.get("intent", "")
        entities = state.get("entities", {})
        missing_fields: List[str] = []

        logger.info("DecisionValidationNode evaluating intent: %s with entities: %s", intent_str, entities)

        # 1. Validation for Appointment Booking
        if intent_str == AgentIntent.APPOINTMENT_BOOKING.value:
            if not entities.get("doctor_name") and not entities.get("department"):
                missing_fields.append("doctor or department")
            if not entities.get("date") and not entities.get("time"):
                missing_fields.append("preferred date or time")

            if missing_fields:
                missing_str = " and ".join(missing_fields)
                clarification_msg = (
                    f"To help you book an appointment, please provide the {missing_str}. "
                    "For example: 'Book an appointment with Dr. Sarah in Cardiology for tomorrow at 10 AM'."
                )
                return {
                    "validation_status": ValidationStatus.MISSING_INFO.value,
                    "missing_fields": missing_fields,
                    "final_response": clarification_msg,
                    "current_step": "validation_clarification_required",
                }

        # 2. Validation for Appointment Check
        elif intent_str == AgentIntent.APPOINTMENT_CHECK.value:
            if not entities.get("entity_id") and not entities.get("doctor_name") and not state.get("user_id"):
                clarification_msg = (
                    "To look up your appointment details, please provide your Patient ID or Appointment ID (e.g., APT-1002)."
                )
                return {
                    "validation_status": ValidationStatus.MISSING_INFO.value,
                    "missing_fields": ["appointment_id_or_patient_id"],
                    "final_response": clarification_msg,
                    "current_step": "validation_clarification_required",
                }

        # 3. Valid parameter state
        return {
            "validation_status": ValidationStatus.VALID.value,
            "missing_fields": [],
            "current_step": "validation_passed",
        }
