"""RAG Service combining Hybrid Retrieval, Context Building, and LLM Prompt Construction."""
import logging
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.rag.context import ContextBuilder
from app.rag.models import RAGContextResult, RAGSearchResult
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are HOPCA, an intelligent and empathetic Hospital Operations & Patient Coordination Agent.\n"
    "Use the provided hospital operational guidelines and reference context below to accurately and professionally "
    "answer patient and staff inquiries. Do not invent hospital policies or medical facts. If the provided context "
    "does not contain the necessary information, state clearly that you do not have that specific information."
)


class RAGService:
    """High-level service orchestrating Hybrid Retrieval and LLM Context Synthesis."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        context_builder: Optional[ContextBuilder] = None,
        db: Optional[AsyncIOMotorDatabase] = None,
        max_context_chars: int = 4000,
    ):
        self.retriever = retriever if retriever is not None else HybridRetriever(db=db)
        self.context_builder = (
            context_builder if context_builder is not None else ContextBuilder(max_context_chars=max_context_chars)
        )

    async def query_knowledge_base(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> RAGContextResult:
        """Execute hybrid search across hospital knowledge base and return formatted context."""
        search_result: RAGSearchResult = await self.retriever.search(
            query=query,
            top_k=top_k,
            domain_filter=domain_filter,
            score_threshold=score_threshold,
        )

        context_result: RAGContextResult = self.context_builder.build_context(
            query=query,
            chunks=search_result.chunks,
        )
        return context_result

    def build_llm_messages(
        self,
        query: str,
        context_result: RAGContextResult,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Construct LLM messages array incorporating system persona, retrieved context, and conversation history."""
        sys_prompt = system_prompt or DEFAULT_RAG_SYSTEM_PROMPT

        system_message_content = (
            f"{sys_prompt}\n\n"
            "--- HOSPITAL KNOWLEDGE CONTEXT ---\n"
            f"{context_result.formatted_context}\n"
            "-----------------------------------"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_message_content}]

        # Append prior conversation history if provided
        if conversation_history:
            for turn in conversation_history:
                messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})

        # Append current user query
        messages.append({"role": "user", "content": query})
        return messages


rag_service = RAGService()
