from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from clinical_rag.config import RAGConfig
from clinical_rag.grounding import GroundingChecker, GroundingResult
from clinical_rag.retrieval.retriever import TwoStageRetriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical information assistant. Answer the question using ONLY
the provided clinical notes. Cite the source note IDs in square brackets when making claims.
If the notes do not contain enough information to answer, say so clearly.
Do not make up clinical information or provide medical advice."""


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    relevant_documents: list[Document]
    answer: str
    grounding: GroundingResult | None
    messages: Annotated[list, add_messages]


def build_rag_graph(
    retriever: TwoStageRetriever,
    config: RAGConfig | None = None,
) -> StateGraph:
    """Build the LangGraph RAG pipeline with grounding checks."""
    config = config or RAGConfig()
    llm = ChatOpenAI(model=config.llm_model, temperature=0)
    grounding_checker = GroundingChecker(config)

    def retrieve(state: RAGState) -> dict:
        question = state["question"]
        docs = retriever.retrieve(question)
        logger.info("Retrieved %d documents for: %s", len(docs), question[:80])
        return {"documents": docs}

    def grade_documents(state: RAGState) -> dict:
        docs = state["documents"]
        question = state["question"]

        prompt = (
            f"Question: {question}\n\n"
            "For each document below, respond YES if it is relevant to answering "
            "the question, NO otherwise. Respond with one word per line.\n\n"
        )
        for i, doc in enumerate(docs):
            prompt += f"Document {i+1}: {doc.page_content[:300]}\n\n"

        response = llm.invoke([HumanMessage(content=prompt)])
        lines = response.content.strip().split("\n")

        relevant: list[Document] = []
        for i, doc in enumerate(docs):
            if i < len(lines) and "YES" in lines[i].upper():
                relevant.append(doc)

        if not relevant and docs:
            relevant = docs[:2]

        logger.info("Graded %d/%d documents as relevant", len(relevant), len(docs))
        return {"relevant_documents": relevant}

    def generate(state: RAGState) -> dict:
        docs = state["relevant_documents"]
        question = state["question"]

        context_parts: list[str] = []
        for doc in docs:
            note_id = doc.metadata.get("note_id", "unknown")
            section = doc.metadata.get("section", "")
            header = f"[{note_id}]"
            if section and section != "FULL_NOTE":
                header += f" ({section})"
            context_parts.append(f"{header}\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Clinical notes:\n\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer with citations to note IDs:"
                )
            ),
        ]

        response = llm.invoke(messages)
        answer = response.content
        logger.info("Generated answer (%d chars)", len(answer))
        return {"answer": answer, "messages": messages + [response]}

    def grounding_check(state: RAGState) -> dict:
        answer = state["answer"]
        docs = state["relevant_documents"]
        result = grounding_checker.check(answer, docs)
        return {"grounding": result}

    def should_refuse(state: RAGState) -> str:
        grounding = state.get("grounding")
        if grounding and not grounding.is_grounded:
            return "refuse"
        return "accept"

    def refuse(state: RAGState) -> dict:
        grounding = state.get("grounding")
        score_str = f"{grounding.score:.2f}" if grounding else "N/A"
        refusal = (
            f"{config.refusal_message}\n\n"
            f"(Grounding score: {score_str}, threshold: {config.grounding_threshold})"
        )
        logger.info("Refusing answer: grounding score %s", score_str)
        return {"answer": refusal}

    def accept(state: RAGState) -> dict:
        return {}

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("grounding_check", grounding_check)
    graph.add_node("refuse", refuse)
    graph.add_node("accept", accept)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("grade_documents", "generate")
    graph.add_edge("generate", "grounding_check")
    graph.add_conditional_edges("grounding_check", should_refuse, {"refuse": "refuse", "accept": "accept"})
    graph.add_edge("refuse", END)
    graph.add_edge("accept", END)

    return graph.compile()
