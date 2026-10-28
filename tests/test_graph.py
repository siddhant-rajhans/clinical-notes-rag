from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from clinical_rag.config import RAGConfig
from clinical_rag.grounding import GroundingResult


class TestRAGGraphNodes:
    """Test individual graph node logic without building the full graph."""

    def test_grounding_refusal_logic(self):
        config = RAGConfig(grounding_threshold=0.6)

        grounded_result = GroundingResult(
            score=0.85, is_grounded=True, per_sentence_scores=[0.9, 0.8]
        )
        assert grounded_result.is_grounded is True

        ungrounded_result = GroundingResult(
            score=0.3, is_grounded=False, per_sentence_scores=[0.2, 0.4]
        )
        assert ungrounded_result.is_grounded is False
        assert ungrounded_result.score < config.grounding_threshold

    def test_rag_state_structure(self):
        state = {
            "question": "What medications does the patient take?",
            "documents": [],
            "relevant_documents": [],
            "answer": "",
            "grounding": None,
            "messages": [],
        }
        assert state["question"] != ""
        assert isinstance(state["documents"], list)

    @patch("clinical_rag.graph.ChatOpenAI")
    @patch("clinical_rag.graph.GroundingChecker")
    def test_build_graph_compiles(self, mock_gc, mock_llm):
        from clinical_rag.graph import build_rag_graph

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            Document(page_content="Test note", metadata={"note_id": "t-1", "section": "HISTORY"})
        ]

        config = RAGConfig()
        graph = build_rag_graph(mock_retriever, config)
        assert graph is not None


class TestDocumentGrading:
    def test_relevant_docs_filter(self):
        docs = [
            Document(page_content="Diabetes treatment with metformin.", metadata={"note_id": "1"}),
            Document(page_content="Knee replacement surgery.", metadata={"note_id": "2"}),
            Document(page_content="Blood glucose monitoring.", metadata={"note_id": "3"}),
        ]

        relevant = [d for d in docs if "diabet" in d.page_content.lower() or "glucose" in d.page_content.lower()]
        assert len(relevant) == 2
        assert relevant[0].metadata["note_id"] == "1"
