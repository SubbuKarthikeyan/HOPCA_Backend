"""Unit and integration tests for Phase 3: LangGraph Agentic Orchestration."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.decision import DecisionValidationNode
from app.agents.graph import HospitalOperationsGraph
from app.agents.intent import IntentClassifierAgent
from app.agents.rag import RAGAgentNode
from app.agents.state import AgentIntent, AgentState, ValidationStatus
from app.agents.tools import ToolsAgentNode
from app.rag.models import RAGContextResult


# ─────────────────────────────────────────────────────────────────────────────
# 1. Intent Classification & Entity Extraction Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_emergency_detection():
    """Verify that acute emergency keywords trigger immediate emergency classification."""
    classifier = IntentClassifierAgent(llm_service=MagicMock())
    intent, confidence, entities = await classifier.classify(
        "Help! The patient is having severe chest pain and shortness of breath!"
    )
    assert intent == AgentIntent.EMERGENCY
    assert confidence >= 0.95
    assert "emergency_trigger" in entities


@pytest.mark.asyncio
async def test_intent_appointment_booking_and_entities():
    """Verify booking intent classification and entity extraction."""
    classifier = IntentClassifierAgent(llm_service=MagicMock())
    intent, confidence, entities = await classifier.classify(
        "Book an appointment with Dr. Sarah Jenkins in Cardiology for tomorrow at 10 AM"
    )
    assert intent == AgentIntent.APPOINTMENT_BOOKING
    assert entities.get("doctor_name") == "Dr. Sarah Jenkins"
    assert entities.get("department") == "cardiology"
    assert entities.get("date") == "tomorrow"
    assert "10" in entities.get("time", "")


@pytest.mark.asyncio
async def test_intent_knowledge_query():
    """Verify hospital guideline and policy classification."""
    classifier = IntentClassifierAgent(llm_service=MagicMock())
    intent, confidence, entities = await classifier.classify(
        "What are the hospital visiting hours for the general ward?"
    )
    assert intent == AgentIntent.KNOWLEDGE_QUERY


# ─────────────────────────────────────────────────────────────────────────────
# 2. Decision & Validation Node Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_decision_node_missing_booking_info():
    """Verify that booking requests without doctor/date trigger a polite clarification prompt."""
    decision_node = DecisionValidationNode()
    state: AgentState = {
        "intent": AgentIntent.APPOINTMENT_BOOKING.value,
        "entities": {},  # Empty entities
        "current_step": "intent_classified",
    }
    result = await decision_node(state)
    assert result["validation_status"] == ValidationStatus.MISSING_INFO.value
    assert len(result["missing_fields"]) > 0
    assert "To help you book an appointment" in result["final_response"]


@pytest.mark.asyncio
async def test_decision_node_valid_booking_info():
    """Verify that complete booking requests pass validation."""
    decision_node = DecisionValidationNode()
    state: AgentState = {
        "intent": AgentIntent.APPOINTMENT_BOOKING.value,
        "entities": {
            "doctor_name": "Dr. Sarah Jenkins",
            "date": "tomorrow",
            "time": "10:00 AM",
        },
        "current_step": "intent_classified",
    }
    result = await decision_node(state)
    assert result["validation_status"] == ValidationStatus.VALID.value
    assert len(result["missing_fields"]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Complete Graph Workflow Execution Tests (Tests 1 - 5)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_graph_test1_knowledge_question_routing():
    """Test 1: Knowledge question -> Intent -> RAG Node -> Grounded response."""
    mock_rag_svc = MagicMock()
    mock_rag_svc.query_knowledge_base = AsyncMock(return_value=RAGContextResult(
        query="visiting policy",
        formatted_context="Visiting hours in General Ward are 9:00 AM to 8:00 PM daily.",
        sources=[{"title": "Hospital Visiting Guidelines", "document_id": "doc_visit"}],
        chunks_used=1,
        has_context=True,
    ))
    mock_rag_svc.build_llm_messages = MagicMock(return_value=[{"role": "user", "content": "test"}])

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=(
        "Visiting hours are from 9:00 AM to 8:00 PM daily in the general ward.",
        "groq",
        "llama-3.3-70b-versatile",
    ))

    graph = HospitalOperationsGraph(llm_service=mock_llm, rag_service=mock_rag_svc)

    input_state: AgentState = {
        "messages": [{"role": "user", "content": "What is the hospital visiting policy?"}],
    }

    final_state = await graph.execute(input_state)
    assert final_state.get("intent") == AgentIntent.KNOWLEDGE_QUERY.value
    assert final_state.get("current_step") == "rag_completed"
    assert "Visiting hours are from 9:00 AM" in final_state.get("final_response")


@pytest.mark.asyncio
async def test_graph_test2_appointment_availability_routing():
    """Test 2: Appointment inquiry -> Intent -> Tools path."""
    mock_llm = MagicMock()
    graph = HospitalOperationsGraph(llm_service=mock_llm)

    input_state: AgentState = {
        "messages": [{"role": "user", "content": "Find available cardiology appointments with Dr. Sarah"}],
    }

    final_state = await graph.execute(input_state)
    assert final_state.get("intent") == AgentIntent.APPOINTMENT_INQUIRY.value
    assert final_state.get("current_step") == "tool_executed"
    assert "available appointment slots" in final_state.get("final_response")


@pytest.mark.asyncio
async def test_graph_test3_missing_booking_information():
    """Test 3: Booking request without info -> Decision asks clarification rather than guessing."""
    mock_llm = MagicMock()
    graph = HospitalOperationsGraph(llm_service=mock_llm)

    input_state: AgentState = {
        "messages": [{"role": "user", "content": "Book me an appointment please"}],
    }

    final_state = await graph.execute(input_state)
    assert final_state.get("intent") == AgentIntent.APPOINTMENT_BOOKING.value
    assert final_state.get("validation_status") == ValidationStatus.MISSING_INFO.value
    assert "please provide the" in final_state.get("final_response")


@pytest.mark.asyncio
async def test_graph_test4_emergency_routing():
    """Test 4: Emergency condition -> Immediate ER hotline response."""
    mock_llm = MagicMock()
    graph = HospitalOperationsGraph(llm_service=mock_llm)

    input_state: AgentState = {
        "messages": [{"role": "user", "content": "The patient collapsed with chest pain!"}],
    }

    final_state = await graph.execute(input_state)
    assert final_state.get("intent") == AgentIntent.EMERGENCY.value
    assert final_state.get("current_step") == "emergency_handled"
    assert "MEDICAL EMERGENCY NOTICE" in final_state.get("final_response")
    assert "911" in final_state.get("final_response")


@pytest.mark.asyncio
async def test_graph_test5_deterministic_completion_no_infinite_loops():
    """Test 5: Graph completes in finite single-pass steps without cycles."""
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=("Hello! How can I help you today?", "groq", "llama-3.3-70b-versatile"))
    graph = HospitalOperationsGraph(llm_service=mock_llm)

    input_state: AgentState = {
        "messages": [{"role": "user", "content": "Hello there!"}],
    }

    final_state = await graph.execute(input_state)
    assert final_state.get("intent") == AgentIntent.GENERAL_CONVERSATION.value
    assert final_state.get("current_step") == "conversation_completed"
    assert final_state.get("final_response") is not None
