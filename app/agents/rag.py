"""RAG agent node utilizing Phase 2 Hybrid RAG service to answer hospital knowledge inquiries."""
import logging
from typing import Any, Dict, Optional

from app.agents.state import AgentState
from app.rag.rag_service import RAGService, rag_service
from app.services.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGAgentNode:
    """Agent node responsible for executing grounded Hybrid RAG queries and response synthesis."""

    def __init__(
        self,
        rag_svc: Optional[RAGService] = None,
        llm_service: Optional[LLMService] = None,
    ):
        self.rag_service = rag_svc if rag_svc is not None else rag_service
        self.llm_service = llm_service if llm_service is not None else LLMService()

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Execute RAG knowledge retrieval and LLM synthesis."""
        messages = state.get("messages", [])
        last_query = ""
        history = []
        for m in messages:
            if m.get("role") == "user":
                last_query = m.get("content", "")
            else:
                history.append(m)

        entities = state.get("entities", {})
        domain_filter = entities.get("department")

        logger.info("RAGAgentNode querying knowledge base for: '%s' (domain: %s)", last_query, domain_filter)

        # 1. Execute Hybrid RAG Query
        context_result = await self.rag_service.query_knowledge_base(
            query=last_query,
            top_k=5,
            domain_filter=domain_filter,
        )

        # 2. Build LLM prompt with grounded context & conversation history
        llm_messages = self.rag_service.build_llm_messages(
            query=last_query,
            context_result=context_result,
            conversation_history=history,
        )

        # 3. Generate response via resilient multi-provider LLM service
        reply, provider, model = await self.llm_service.generate(
            messages=llm_messages,
            temperature=0.3,  # Low temperature for factual precision
        )

        # Append source footnotes if available and not already in text
        if context_result.has_context and context_result.sources:
            unique_titles = list({s.get("title") for s in context_result.sources if s.get("title")})
            if unique_titles and "Source:" not in reply and "References:" not in reply:
                sources_str = ", ".join(unique_titles[:3])
                reply = f"{reply}\n\n*References: {sources_str}*"

        return {
            "retrieved_context": context_result.formatted_context,
            "sources": context_result.sources,
            "final_response": reply,
            "current_step": "rag_completed",
        }
