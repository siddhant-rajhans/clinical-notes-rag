from __future__ import annotations

import logging

from langchain_core.documents import Document

from clinical_rag.config import RAGConfig
from clinical_rag.index import FAISSIndex
from clinical_rag.retrieval.reranker import ClinicalReranker

logger = logging.getLogger(__name__)


class TwoStageRetriever:
    """Two-stage retrieval: FAISS dense search followed by cross-encoder reranking."""

    def __init__(
        self,
        index: FAISSIndex,
        config: RAGConfig | None = None,
    ):
        self.config = config or RAGConfig()
        self.index = index
        self.reranker = ClinicalReranker(config)

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve and rerank documents for a query.

        Stage 1: FAISS dense retrieval returns top_k_retrieval candidates.
        Stage 2: Cross-encoder reranks to top_k_rerank final documents.
        """
        stage1_results = self.index.search(query, top_k=self.config.top_k_retrieval)
        stage1_docs = [doc for doc, _score in stage1_results]

        logger.info(
            "Stage 1: retrieved %d candidates (top score=%.3f)",
            len(stage1_docs),
            stage1_results[0][1] if stage1_results else 0.0,
        )

        reranked = self.reranker.rerank(
            query, stage1_docs, top_k=self.config.top_k_rerank
        )
        final_docs = [doc for doc, _score in reranked]

        logger.info(
            "Stage 2: reranked to %d documents (top score=%.3f)",
            len(final_docs),
            reranked[0][1] if reranked else 0.0,
        )

        return final_docs
