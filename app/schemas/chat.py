from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's message to the agent")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Previous conversation turns")
    system_prompt: Optional[str] = Field(
        default="You are a helpful, accurate, and empathetic Hospital Operations & Patient Coordination AI assistant.",
        description="Optional system prompt to guide agent persona",
    )
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    preferred_provider: Optional[str] = Field(
        default=None,
        description="Optional preferred provider (groq, gemini, mistral, deepseek)",
    )
    model: Optional[str] = Field(default=None, description="Optional specific model override")
    stream: bool = Field(default=False, description="Whether to stream response")

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty or whitespace")
        return v.strip()


class ChatResponse(BaseModel):
    response: str
    provider: str
    model: str


class StreamChunk(BaseModel):
    chunk: str
    provider: Optional[str] = None
    model: Optional[str] = None
    is_final: bool = False
    error: Optional[str] = None
