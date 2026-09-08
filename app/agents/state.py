"""Shared state schema and data contracts for Hospital Operations LangGraph agents."""
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class AgentIntent(str, Enum):
    """Supported hospital intent categories."""
    KNOWLEDGE_QUERY = "KNOWLEDGE_QUERY"
    APPOINTMENT_INQUIRY = "APPOINTMENT_INQUIRY"
    APPOINTMENT_BOOKING = "APPOINTMENT_BOOKING"
    APPOINTMENT_CHECK = "APPOINTMENT_CHECK"
    INSURANCE_INQUIRY = "INSURANCE_INQUIRY"
    PATIENT_COORDINATION = "PATIENT_COORDINATION"
    EMERGENCY = "EMERGENCY"
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"


class ValidationStatus(str, Enum):
    """Status outcomes from Decision / Validation node."""
    VALID = "VALID"
    MISSING_INFO = "MISSING_INFO"
    UNAUTHORIZED = "UNAUTHORIZED"
    INVALID_OPERATION = "INVALID_OPERATION"


class AgentState(TypedDict, total=False):
    """LangGraph-compatible TypedDict representing the agent workflow execution state."""
    messages: List[Dict[str, Any]]
    user_id: Optional[str]
    session_id: Optional[str]
    intent: Optional[str]
    intent_confidence: float
    entities: Dict[str, Any]
    retrieved_context: Optional[str]
    sources: List[Dict[str, str]]
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    validation_status: Optional[str]
    missing_fields: List[str]
    requires_human_approval: bool
    approval_status: Optional[str]
    final_response: Optional[str]
    current_step: str
    error: Optional[str]


class AgentStateModel(BaseModel):
    """Pydantic model representation of AgentState for serialization and validation."""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    intent: Optional[AgentIntent] = None
    intent_confidence: float = 0.0
    entities: Dict[str, Any] = Field(default_factory=dict)
    retrieved_context: Optional[str] = None
    sources: List[Dict[str, str]] = Field(default_factory=list)
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    validation_status: Optional[ValidationStatus] = None
    missing_fields: List[str] = Field(default_factory=list)
    requires_human_approval: bool = False
    approval_status: Optional[str] = None
    final_response: Optional[str] = None
    current_step: str = "init"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to standard dict."""
        return self.model_dump()
