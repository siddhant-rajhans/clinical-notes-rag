from __future__ import annotations

import logging

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from clinical_rag.config import RAGConfig

logger = logging.getLogger(__name__)


class ClinicalReranker:
    """Cross-encoder reranker for clinical document retrieval."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading reranker: %s", self.config.reranker_model)
            self._model = CrossEncoder(self.config.reranker_model)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Rerank documents by cross-encoder relevance to the query."""
        if not documents:
            return []

        top_k = top_k or self.config.top_k_rerank
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [(doc, float(score)) for doc, score in scored[:top_k]]
