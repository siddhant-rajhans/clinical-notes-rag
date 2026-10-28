from __future__ import annotations

import logging

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from clinical_rag.config import RAGConfig

logger = logging.getLogger(__name__)


class ClinicalEmbedder:
    """BioClinicalBERT embedding wrapper using sentence-transformers."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.config.embedding_model)
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Encode a list of texts into normalized embeddings."""
        all_embeddings: list[np.ndarray] = []
        batch_size = self.config.embedding_batch_size

        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(batches, desc="Embedding", unit="batch")

        for start in batches:
            batch = texts[start : start + batch_size]
            emb = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(emb)

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_documents(
        self, documents: list[Document], show_progress: bool = True
    ) -> np.ndarray:
        """Encode document page_content into normalized embeddings."""
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts, show_progress=show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        return self.embed_texts([query], show_progress=False)[0]
