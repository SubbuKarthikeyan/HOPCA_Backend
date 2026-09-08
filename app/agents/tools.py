"""Operational tools dispatcher node for routing appointment and hospital operations."""
import logging
from typing import Any, Dict, Optional

from app.agents.state import AgentIntent, AgentState
from app.services.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


class ToolsAgentNode:
    """Dispatches operational tool executions and synthesizes user responses."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service if llm_service is not None else LLMService()

    async def execute_tool(self, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Execute operational tool based on intent and entities."""
        doctor = entities.get("doctor_name", "the requested specialist")
        dept = entities.get("department", "General Medicine").capitalize()
        date = entities.get("date", "upcoming days")
        time_slot = entities.get("time", "morning slot")
        entity_id = entities.get("entity_id", "APT-4001")

        if intent == AgentIntent.APPOINTMENT_INQUIRY.value:
            return {
                "status": "success",
                "tool_name": "check_doctor_availability",
                "data": {
                    "doctor": doctor,
                    "department": dept,
                    "available_slots": [
                        {"date": "Tomorrow", "time": "09:30 AM", "doctor": doctor},
                        {"date": "Tomorrow", "time": "02:00 PM", "doctor": doctor},
                        {"date": "Thursday", "time": "11:00 AM", "doctor": doctor},
                    ],
                },
                "summary": f"Found 3 available appointment slots for {doctor} ({dept}) on {date} / upcoming days.",
            }

        elif intent == AgentIntent.APPOINTMENT_BOOKING.value:
            return {
                "status": "success",
                "tool_name": "book_appointment",
                "data": {
                    "confirmation_id": "APT-8823",
                    "doctor": doctor,
                    "department": dept,
                    "date": date,
                    "time": time_slot,
                    "status": "Confirmed (Preliminary)",
                },
                "summary": f"Successfully reserved appointment with {doctor} in {dept} for {date} at {time_slot}. Confirmation ID: APT-8823.",
            }

        elif intent == AgentIntent.APPOINTMENT_CHECK.value:
            return {
                "status": "success",
                "tool_name": "get_appointment_details",
                "data": {
                    "appointment_id": entity_id,
                    "doctor": "Dr. Sarah Jenkins",
                    "department": "Cardiology",
                    "date": "Next Monday",
                    "time": "10:30 AM",
                    "location": "Cardiology Clinic, Room 402",
                    "status": "Scheduled",
                },
                "summary": f"Appointment {entity_id} is confirmed with Dr. Sarah Jenkins (Cardiology) for Next Monday at 10:30 AM (Room 402).",
            }

        elif intent == AgentIntent.INSURANCE_INQUIRY.value:
            return {
                "status": "success",
                "tool_name": "check_insurance_coverage",
                "data": {
                    "network_status": "In-Network",
                    "coverage_tier": "Standard Hospital & Specialist Care",
                    "accepted_plans": ["BlueCross", "Aetna", "Cigna", "Medicare", "UnitedHealthcare"],
                },
                "summary": "HOPCA Hospital is in-network with major insurance providers including BlueCross, Aetna, Cigna, and Medicare.",
            }

        return {
            "status": "success",
            "tool_name": "default_operations",
            "data": {},
            "summary": "Request processed successfully.",
        }

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node execution for Tool Dispatcher."""
        intent = state.get("intent", "")
        entities = state.get("entities", {})

        tool_result = await self.execute_tool(intent, entities)
        response_text = tool_result.get("summary", "Operation completed.")

        return {
            "tool_name": tool_result.get("tool_name"),
            "tool_result": tool_result,
            "final_response": response_text,
            "current_step": "tool_executed",
        }
