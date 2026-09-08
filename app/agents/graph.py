"""LangGraph Multi-Agent Workflow Orchestrator for Hospital Operations."""
import logging
from typing import Any, Dict, List, Optional

from app.agents.decision import DecisionValidationNode
from app.agents.intent import IntentClassifierAgent
from app.agents.rag import RAGAgentNode
from app.agents.state import AgentIntent, AgentState, ValidationStatus
from app.agents.tools import ToolsAgentNode
from app.rag.rag_service import RAGService
from app.services.llm.llm_service import LLMService

logger = logging.getLogger(__name__)

EMERGENCY_RESPONSE_TEMPLATE = (
    "🚨 **MEDICAL EMERGENCY NOTICE**\n\n"
    "If you or the patient are experiencing life-threatening symptoms (such as severe chest pain, "
    "difficulty breathing, uncontrollable bleeding, or loss of consciousness), **please call 911 immediately** "
    "or proceed to the nearest Emergency Room.\n\n"
    "**HOPCA Emergency Department:**\n"
    "📍 Location: Ground Floor, Emergency Entrance (Gate 1)\n"
    "📞 24/7 ER Hotline: **(555) 911-ER-00** / Ext. 9110"
)


class HospitalOperationsGraph:
    """Orchestrates multi-agent routing between Intent, RAG, Decision, and Operational Tool nodes."""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        rag_service: Optional[RAGService] = None,
    ):
        self.llm_service = llm_service if llm_service is not None else LLMService()
        self.intent_node = IntentClassifierAgent(llm_service=self.llm_service)
        self.rag_node = RAGAgentNode(rag_svc=rag_service, llm_service=self.llm_service)
        self.decision_node = DecisionValidationNode()
        self.tools_node = ToolsAgentNode(llm_service=self.llm_service)

    async def emergency_node(self, state: AgentState) -> Dict[str, Any]:
        """Emergency response node providing immediate contact and safety instructions."""
        return {
            "final_response": EMERGENCY_RESPONSE_TEMPLATE,
            "current_step": "emergency_handled",
        }

    async def conversation_node(self, state: AgentState) -> Dict[str, Any]:
        """General conversational node handling greetings and pleasantries."""
        messages = state.get("messages", [])
        last_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_msg = m.get("content", "")
                break

        system_prompt = (
            "You are HOPCA, the Hospital Operations & Patient Coordination AI assistant. "
            "Respond warmly, professionally, and concisely to the user. "
            "Ask how you can assist them with appointments, hospital policies, doctors, or general inquiries."
        )
        reply, _, _ = await self.llm_service.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": last_msg or "Hello"},
            ],
            temperature=0.7,
        )
        return {
            "final_response": reply,
            "current_step": "conversation_completed",
        }

    async def route_after_intent(self, state: AgentState) -> str:
        """Determine next node following intent classification."""
        intent = state.get("intent", AgentIntent.KNOWLEDGE_QUERY.value)

        if intent == AgentIntent.EMERGENCY.value:
            return "emergency_node"
        elif intent == AgentIntent.GENERAL_CONVERSATION.value:
            return "conversation_node"
        elif intent in (AgentIntent.APPOINTMENT_BOOKING.value, AgentIntent.APPOINTMENT_CHECK.value):
            return "decision_node"
        elif intent in (AgentIntent.APPOINTMENT_INQUIRY.value, AgentIntent.INSURANCE_INQUIRY.value, AgentIntent.PATIENT_COORDINATION.value):
            return "tools_node"
        else:
            return "rag_node"

    async def route_after_decision(self, state: AgentState) -> str:
        """Determine next node following decision validation."""
        val_status = state.get("validation_status")
        if val_status == ValidationStatus.VALID.value:
            return "tools_node"
        # If missing info or unauthorized, terminal response is already formatted
        return "END"

    async def execute(self, state: AgentState) -> AgentState:
        """Execute the complete deterministic multi-agent state graph."""
        current_state: AgentState = dict(state)

        # 1. Step 1: Intent Classification
        logger.info("Executing Graph: Step 1 -> Intent Classification")
        intent_updates = await self.intent_node(current_state)
        current_state.update(intent_updates)

        # 2. Step 2: Intent Routing
        next_node = await self.route_after_intent(current_state)
        logger.info("Routing from intent '%s' -> Node '%s'", current_state.get("intent"), next_node)

        if next_node == "emergency_node":
            res = await self.emergency_node(current_state)
            current_state.update(res)

        elif next_node == "conversation_node":
            res = await self.conversation_node(current_state)
            current_state.update(res)

        elif next_node == "rag_node":
            res = await self.rag_node(current_state)
            current_state.update(res)

        elif next_node == "decision_node":
            decision_res = await self.decision_node(current_state)
            current_state.update(decision_res)

            decision_next = await self.route_after_decision(current_state)
            if decision_next == "tools_node":
                tool_res = await self.tools_node(current_state)
                current_state.update(tool_res)

        elif next_node == "tools_node":
            tool_res = await self.tools_node(current_state)
            current_state.update(tool_res)

        logger.info("Graph execution reached terminal state. Step: %s", current_state.get("current_step"))
        return current_state


# Global singleton instance
hospital_agent_graph = HospitalOperationsGraph()
