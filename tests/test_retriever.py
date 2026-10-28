from unittest.mock import MagicMock, patch

import numpy as np
from langchain_core.documents import Document

from clinical_rag.config import RAGConfig
from clinical_rag.index import FAISSIndex
from clinical_rag.retrieval.retriever import TwoStageRetriever


def _make_docs(n: int) -> list[Document]:
    return [
        Document(
            page_content=f"Clinical note {i} about patient treatment.",
            metadata={"note_id": f"test-{i}", "section": "HISTORY"},
        )
        for i in range(n)
    ]


class TestTwoStageRetriever:
    @patch("clinical_rag.retrieval.retriever.ClinicalReranker")
    def test_retrieve_calls_both_stages(self, mock_reranker_cls):
        docs = _make_docs(5)
        mock_index = MagicMock(spec=FAISSIndex)
        mock_index.search.return_value = [(doc, 0.9 - i * 0.1) for i, doc in enumerate(docs)]

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [(docs[2], 0.95), (docs[0], 0.90)]
        mock_reranker_cls.return_value = mock_reranker

        config = RAGConfig(top_k_retrieval=5, top_k_rerank=2)
        retriever = TwoStageRetriever(mock_index, config)
        results = retriever.retrieve("test query")

        mock_index.search.assert_called_once_with("test query", top_k=5)
        mock_reranker.rerank.assert_called_once()
        assert len(results) == 2
        assert results[0].metadata["note_id"] == "test-2"

    @patch("clinical_rag.retrieval.retriever.ClinicalReranker")
    def test_empty_index_returns_empty(self, mock_reranker_cls):
        mock_index = MagicMock(spec=FAISSIndex)
        mock_index.search.return_value = []

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []
        mock_reranker_cls.return_value = mock_reranker

        retriever = TwoStageRetriever(mock_index)
        results = retriever.retrieve("query with no matches")
        assert results == []


class TestFAISSIndexSearch:
    def test_search_returns_scored_documents(self):
        docs = _make_docs(10)
        dim = 8
        embeddings = np.random.randn(10, dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        import faiss

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        mock_embedder = MagicMock()
        query_vec = np.random.randn(dim).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        mock_embedder.embed_query.return_value = query_vec

        faiss_index = FAISSIndex(index=index, documents=docs, embedder=mock_embedder)
        results = faiss_index.search("test", top_k=3)

        assert len(results) == 3
        for doc, score in results:
            assert isinstance(doc, Document)
            assert -1.0 <= score <= 1.0
